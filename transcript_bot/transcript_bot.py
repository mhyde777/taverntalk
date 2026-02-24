import asyncio
import json
import os
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import discord
import discord.sinks
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import ollama as ollama_client

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("transcript_bot")

# -------------------------------------------------
# Environment / Config
# -------------------------------------------------

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set in the environment or .env file.")

GUILD_IDS_RAW = os.getenv("GUILD_ID", "")

SUMMARY_CHANNEL_ID_RAW = os.getenv("SUMMARY_CHANNEL_ID", "0")
try:
    SUMMARY_CHANNEL_ID: Optional[int] = int(SUMMARY_CHANNEL_ID_RAW) or None
except ValueError:
    SUMMARY_CHANNEL_ID = None

TRANSCRIPT_DIR = os.getenv("TRANSCRIPT_DIR", "./transcripts")
CAMPAIGN_NAME = os.getenv("CAMPAIGN_NAME", "TavernTalk")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CHARACTERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "characters")


def parse_guild_ids(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    guild_ids: List[int] = []
    for p in parts:
        try:
            guild_ids.append(int(p))
        except ValueError:
            logger.warning("Skipping invalid guild id: %r", p)
    return guild_ids


GUILD_IDS: List[int] = parse_guild_ids(GUILD_IDS_RAW)
if not GUILD_IDS:
    logger.warning(
        "GUILD_ID is not set. "
        "Slash commands will be registered globally (can take up to an hour)."
    )

# -------------------------------------------------
# Whisper model + thread pool (loaded once at startup)
# -------------------------------------------------

logger.info("Loading Whisper model '%s' on CPU (int8)...", WHISPER_MODEL_SIZE)
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
executor = ThreadPoolExecutor(max_workers=2)
logger.info("Whisper model loaded.")

# -------------------------------------------------
# Character map persistence
# -------------------------------------------------

def load_character_map(guild_id: int) -> Dict[int, str]:
    path = os.path.join(CHARACTERS_DIR, f"{guild_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
    return {}


def save_character_map(guild_id: int, mapping: Dict[int, str]) -> None:
    os.makedirs(CHARACTERS_DIR, exist_ok=True)
    path = os.path.join(CHARACTERS_DIR, f"{guild_id}.json")
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)

# -------------------------------------------------
# Session / State dataclasses
# -------------------------------------------------

@dataclass
class VoiceSession:
    guild_id: int
    voice_client: discord.VoiceClient
    text_channel_id: int
    started_by_id: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True


class VoiceSessionManager:
    def __init__(self) -> None:
        self.sessions: Dict[int, VoiceSession] = {}

    def get(self, guild_id: int) -> Optional[VoiceSession]:
        return self.sessions.get(guild_id)

    def add(self, session: VoiceSession) -> None:
        self.sessions[session.guild_id] = session

    def stop(self, guild_id: int) -> Optional[VoiceSession]:
        session = self.sessions.get(guild_id)
        if session and session.active:
            session.active = False
            return session
        return None


voice_sessions = VoiceSessionManager()


class SessionManager:
    """Tracks character name mappings per guild, with JSON persistence."""

    def __init__(self) -> None:
        # guild_id -> {user_id: character_name}; loaded lazily
        self.character_map: Dict[int, Dict[int, str]] = {}

    def _ensure_loaded(self, guild_id: int) -> None:
        if guild_id not in self.character_map:
            self.character_map[guild_id] = load_character_map(guild_id)

    def set_character(self, guild_id: int, user_id: int, character_name: str) -> None:
        self._ensure_loaded(guild_id)
        self.character_map[guild_id][user_id] = character_name
        save_character_map(guild_id, self.character_map[guild_id])

    def get_character(self, guild_id: int, user_id: int) -> Optional[str]:
        self._ensure_loaded(guild_id)
        return self.character_map.get(guild_id, {}).get(user_id)


session_manager = SessionManager()

# -------------------------------------------------
# Intents and Bot
# -------------------------------------------------

intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
intents.messages = True
intents.message_content = True


class TranscriptBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self) -> None:
        if GUILD_IDS:
            logger.info("Syncing application commands to guilds: %s", GUILD_IDS)
            for gid in GUILD_IDS:
                guild = discord.Object(id=gid)
                try:
                    self.tree.copy_global_to(guild=guild)
                    synced = await self.tree.sync(guild=guild)
                    logger.info("Synced %d commands to guild %d.", len(synced), gid)
                except Exception:
                    logger.exception("Failed to sync commands for guild %d", gid)
        else:
            logger.info("Syncing global application commands.")
            try:
                synced = await self.tree.sync()
                logger.info("Synced %d global commands.", len(synced))
            except Exception:
                logger.exception("Failed to sync global commands")

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        logger.info("Connected to %d guild(s).", len(self.guilds))


bot = TranscriptBot()

# -------------------------------------------------
# Audio helpers
# -------------------------------------------------

def _transcribe_wav_sync(wav_path: str) -> str:
    """Run Whisper transcription synchronously (called in thread pool)."""
    segments, _ = whisper_model.transcribe(wav_path, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()


async def transcribe_wav(wav_path: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _transcribe_wav_sync, wav_path)


async def convert_to_16k_mono(raw_bytes: bytes) -> str:
    """Write raw WAV bytes to disk, resample to 16 kHz mono, return output path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
        src.write(raw_bytes)
        src_path = src.name

    dst_path = src_path.replace(".wav", "_16k.wav")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", src_path,
        "-ar", "16000", "-ac", "1", dst_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    os.unlink(src_path)
    return dst_path

# -------------------------------------------------
# Ollama summary
# -------------------------------------------------

async def generate_summary(transcript_text: str, session_info: str) -> str:
    prompt = (
        "You are a dungeon master's scribe summarizing a Dungeons & Dragons session. "
        "Based on the following transcript, write a narrative summary that covers:\n"
        "- What happened in the session (key plot points)\n"
        "- Important decisions the players made\n"
        "- Notable moments, encounters, or role-play highlights\n"
        "- Any unresolved threads or cliffhangers\n\n"
        f"Session info: {session_info}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Summary:"
    )
    try:
        client = ollama_client.AsyncClient(host=OLLAMA_HOST)
        response = await client.generate(model=OLLAMA_MODEL, prompt=prompt)
        return response["response"].strip()
    except Exception as e:
        logger.exception("Ollama summarization failed: %s", e)
        return f"[Summary unavailable: {e}]"

# -------------------------------------------------
# Transcript file (Obsidian Markdown)
# -------------------------------------------------

async def save_transcript_file(
    voice_channel_name: str,
    participants: List[str],
    transcript_lines: List[str],
    summary: str,
    started_at: datetime,
    ended_at: datetime,
) -> str:
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    date_str = started_at.strftime("%Y-%m-%d")
    time_str = started_at.strftime("%H-%M")
    filename = f"{date_str}_{time_str}_{CAMPAIGN_NAME.replace(' ', '_')}.md"
    filepath = os.path.join(TRANSCRIPT_DIR, filename)

    frontmatter = (
        "---\n"
        f"date: {date_str}\n"
        f"campaign: {CAMPAIGN_NAME}\n"
        f"voice_channel: {voice_channel_name}\n"
        "participants:\n"
        + "".join(f"  - {p}\n" for p in participants)
        + "---\n"
    )

    duration_min = int((ended_at - started_at).total_seconds() // 60)
    header = (
        f"# {CAMPAIGN_NAME} — Session {date_str}\n\n"
        f"**Started:** {started_at.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Ended:** {ended_at.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        f"**Duration:** ~{duration_min} min  \n"
        f"**Participants:** {', '.join(participants)}\n\n"
    )

    summary_section = f"## Session Summary\n\n{summary}\n\n"
    transcript_section = "## Full Transcript\n\n" + "\n\n".join(transcript_lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(frontmatter + "\n" + header + summary_section + transcript_section)

    return filepath

# -------------------------------------------------
# Git auto-push
# -------------------------------------------------

async def git_push_transcript(filepath: str) -> None:
    """Stage, commit, and push a transcript file to the remote git repo."""
    repo_dir = os.path.abspath(TRANSCRIPT_DIR)
    filename = os.path.basename(filepath)

    async def run(*args) -> int:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=repo_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0 and stderr:
            logger.warning("git %s: %s", args[1], stderr.decode().strip())
        return proc.returncode

    await run("git", "add", filepath)
    rc = await run("git", "commit", "-m", f"Add transcript: {filename}")
    if rc != 0:
        logger.warning("git commit returned %d — nothing to commit?", rc)
        return
    rc = await run("git", "push")
    if rc == 0:
        logger.info("Pushed transcript '%s' to remote.", filename)
    else:
        logger.error("git push failed for '%s'.", filename)

# -------------------------------------------------
# Chunked Discord message posting
# -------------------------------------------------

async def post_chunked(channel: discord.TextChannel, text: str, limit: int = 1900) -> None:
    if len(text) <= limit:
        await channel.send(text)
        return
    lines = text.split("\n\n")
    chunk = ""
    for line in lines:
        block = line + "\n\n"
        if len(chunk) + len(block) > limit:
            await channel.send(chunk)
            chunk = ""
        chunk += block
    if chunk:
        await channel.send(chunk)

# -------------------------------------------------
# Recording finished handler
# -------------------------------------------------

async def handle_recording_finished(guild_id: int, sink: discord.sinks.Sink) -> None:
    ended_at = datetime.now(timezone.utc)
    session = voice_sessions.get(guild_id)
    if not session:
        return

    guild = bot.get_guild(guild_id)
    text_channel = guild.get_channel(session.text_channel_id) if guild else None
    if text_channel is None:
        return

    await text_channel.send("Recording ended. Transcribing audio — this may take a few minutes...")

    transcript_lines: List[str] = []
    participants: List[str] = []

    for user_id, audio_list in sink.audio_data.items():
        member = guild.get_member(user_id) if guild else None
        discord_name = member.display_name if member else f"User {user_id}"
        char_name = session_manager.get_character(guild_id, user_id)
        speaker_label = f"{char_name} ({discord_name})" if char_name else discord_name
        participants.append(speaker_label)

        combined_bytes = b""
        for audio in audio_list:
            audio.file.seek(0)
            combined_bytes += audio.file.read()

        if not combined_bytes:
            transcript_lines.append(f"**{speaker_label}**: [no audio]")
            continue

        try:
            wav_16k_path = await convert_to_16k_mono(combined_bytes)
            text = await transcribe_wav(wav_16k_path)
            os.unlink(wav_16k_path)
        except Exception as e:
            logger.exception("Transcription failed for %s: %s", speaker_label, e)
            text = f"[transcription error: {e}]"

        transcript_lines.append(f"**{speaker_label}**: {text}")

    full_transcript = "\n\n".join(transcript_lines)
    await post_chunked(text_channel, "**Transcript:**\n\n" + full_transcript)

    voice_channel = None
    if guild and session.voice_client:
        voice_channel = session.voice_client.channel
    vc_name = voice_channel.name if voice_channel else "unknown"

    summary_channel = None
    if SUMMARY_CHANNEL_ID and guild:
        summary_channel = guild.get_channel(SUMMARY_CHANNEL_ID)

    session_info = (
        f"Campaign: {CAMPAIGN_NAME}, Channel: #{vc_name}, "
        f"Date: {ended_at.strftime('%Y-%m-%d')}"
    )

    placeholder = None
    if summary_channel:
        placeholder = await summary_channel.send(
            f"**{CAMPAIGN_NAME} — Session {ended_at.strftime('%Y-%m-%d')}**\n"
            "Generating summary... (this may take several minutes on slow hardware)"
        )

    summary = await generate_summary(full_transcript, session_info)

    filepath = await save_transcript_file(
        vc_name, participants, transcript_lines, summary,
        session.started_at, ended_at,
    )
    await text_channel.send(f"Transcript saved to: `{filepath}`")
    await git_push_transcript(filepath)

    if summary_channel and placeholder:
        summary_msg = (
            f"**{CAMPAIGN_NAME} — Session {ended_at.strftime('%Y-%m-%d')}**\n\n"
            f"**Participants:** {', '.join(participants)}\n\n"
            f"**Summary:**\n{summary}"
        )
        await placeholder.edit(content=summary_msg[:2000])

    if session.voice_client and session.voice_client.is_connected():
        await session.voice_client.disconnect(force=True)

# -------------------------------------------------
# Slash commands
# -------------------------------------------------

@bot.tree.command(name="ping", description="Check if the bot is alive.")
async def ping(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("Pong.", ephemeral=True)


@bot.tree.command(
    name="start",
    description="Start live transcription in your current voice channel.",
)
async def start_command(interaction: discord.Interaction) -> None:
    guild = interaction.guild
    user = interaction.user

    if guild is None or not isinstance(user, discord.Member):
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return

    if not user.voice or not user.voice.channel:
        await interaction.response.send_message(
            "You must be in a voice channel to start transcription.", ephemeral=True
        )
        return

    existing = voice_sessions.get(guild.id)
    if existing and existing.active and existing.voice_client.is_connected():
        await interaction.response.send_message(
            "Transcription is already active in this server.", ephemeral=True
        )
        return

    voice_channel = user.voice.channel
    text_channel = interaction.channel

    if not isinstance(text_channel, discord.TextChannel):
        await interaction.response.send_message(
            "This command must be used in a text channel.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=False)

    try:
        voice_client = await voice_channel.connect()
    except discord.ClientException:
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client is None:
            await interaction.followup.send("Failed to connect to the voice channel.")
            return

    sink = discord.sinks.WaveSink()

    def finished_callback(sink: discord.sinks.Sink, *args) -> None:
        bot.loop.create_task(handle_recording_finished(guild.id, sink))

    voice_client.start_recording(sink, finished_callback, None)

    session = VoiceSession(
        guild_id=guild.id,
        voice_client=voice_client,
        text_channel_id=text_channel.id,
        started_by_id=user.id,
    )
    voice_sessions.add(session)

    await interaction.followup.send(
        f"Started transcription in {voice_channel.mention}. "
        "Use `/stop` to end and post the transcript."
    )


@bot.tree.command(
    name="stop",
    description="Stop the current transcription and post the transcript.",
)
async def stop_command(interaction: discord.Interaction) -> None:
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return

    session = voice_sessions.get(guild.id)
    if not session or not session.active:
        await interaction.response.send_message(
            "No active transcription in this server.", ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=False)

    vc = session.voice_client
    if vc and vc.is_connected():
        vc.stop_recording()  # triggers finished_callback

    voice_sessions.stop(guild.id)

    await interaction.followup.send(
        "Stopping transcription. I will post the transcript here when it is ready."
    )


@bot.tree.command(
    name="status",
    description="Show the status of the current transcription session.",
)
async def status_command(interaction: discord.Interaction) -> None:
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return

    session = voice_sessions.get(guild.id)
    if not session or not session.active:
        await interaction.response.send_message(
            "No active transcription session in this server.", ephemeral=True
        )
        return

    vc = session.voice_client
    vc_name = vc.channel.name if vc and vc.channel else "unknown"
    started_at_str = session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (
        f"Transcription is active.\n"
        f"Voice channel: **{vc_name}**\n"
        f"Text channel: <#{session.text_channel_id}>\n"
        f"Started by: <@{session.started_by_id}>\n"
        f"Started at: {started_at_str}"
    )
    await interaction.response.send_message(msg, ephemeral=True)


@bot.tree.command(
    name="set_character",
    description="Associate a Discord user with a character name for transcripts.",
)
@app_commands.describe(
    character_name="The character name to associate with this user.",
    user="The Discord user to set the character for (defaults to yourself).",
)
async def set_character_command(
    interaction: discord.Interaction,
    character_name: str,
    user: Optional[discord.Member] = None,
) -> None:
    guild = interaction.guild
    if guild is None:
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return

    target = user or interaction.user
    session_manager.set_character(guild.id, target.id, character_name)

    if target.id == interaction.user.id:
        msg = f"Set your character name to: **{character_name}**"
    else:
        msg = f"Set character name for {target.mention} to: **{character_name}**"

    await interaction.response.send_message(msg, ephemeral=True)

# -------------------------------------------------
# Prefix command: manual guild sync
# -------------------------------------------------

@bot.command(name="sync_guild", help="Manually re-sync slash commands for this guild.")
@commands.has_guild_permissions(administrator=True)
async def sync_guild(ctx: commands.Context) -> None:
    guild = ctx.guild
    if guild is None:
        await ctx.send("This command can only be used in a server.")
        return

    logger.info("Manually syncing commands for guild %d.", guild.id)
    try:
        bot.tree.copy_global_to(guild=guild)
        synced = await bot.tree.sync(guild=guild)
        await ctx.send(f"Synced {len(synced)} commands to this guild.")
    except Exception as exc:
        logger.exception("Manual sync failed for guild %d: %s", guild.id, exc)
        await ctx.send(f"Sync failed: {exc!r}")

# -------------------------------------------------
# Main
# -------------------------------------------------

def main() -> None:
    logger.info("Starting Transcript Bot.")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
