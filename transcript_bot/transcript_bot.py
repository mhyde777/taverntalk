import asyncio
import io
import json
import os
import tempfile
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import discord
import discord.sinks
from discord.ext import commands
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

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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
CHUNK_INTERVAL = int(os.getenv("CHUNK_INTERVAL", "60"))  # seconds between mid-session transcription passes


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
# Custom sink with drainable pending buffer
# -------------------------------------------------

class ChunkedWaveSink(discord.sinks.WaveSink):
    """WaveSink that maintains a separate thread-safe drainable buffer
    so the background task can pull audio in chunks without touching
    the main audio_data that discord.py manages."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: Dict[int, io.BytesIO] = {}

    def write(self, data: bytes, user: int) -> None:
        with self._lock:
            if user not in self._pending:
                self._pending[user] = io.BytesIO()
            self._pending[user].write(data)
        super().write(data, user)

    def drain_pending(self) -> Dict[int, bytes]:
        """Atomically return all pending audio bytes per user and reset the buffer."""
        with self._lock:
            result: Dict[int, bytes] = {}
            for user_id, buf in self._pending.items():
                buf.seek(0)
                data = buf.read()
                if data:
                    result[user_id] = data
            self._pending.clear()
            return result

# -------------------------------------------------
# Session / State dataclasses
# -------------------------------------------------

@dataclass
class VoiceSession:
    guild_id: int
    voice_client: discord.VoiceClient
    text_channel_id: int
    started_by_id: int
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    active: bool = True
    voice_channel_name: Optional[str] = None
    # Incremental transcription state
    transcript_chunks: Dict[int, List[str]] = field(default_factory=dict)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    chunk_task: Optional[asyncio.Task] = field(default=None)


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
        super().__init__(command_prefix="!", intents=intents, auto_sync_commands=True)

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        logger.info("Connected to %d guild(s).", len(self.guilds))
        if GUILD_IDS:
            # Purge any stale globally-registered commands so guild commands are the only ones visible.
            await self.http.bulk_upsert_global_commands(self.application_id, [])
            logger.info("Cleared global application commands.")
        await self.sync_commands(delete_existing=True)


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
    """Resample a WAV file (bytes) to 16 kHz mono, return output path."""
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


async def convert_pcm_to_16k_mono(pcm_bytes: bytes) -> str:
    """Resample raw 48 kHz stereo s16le PCM (from Discord) to 16 kHz mono WAV."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as src:
        src.write(pcm_bytes)
        src_path = src.name

    dst_path = src_path.replace(".raw", "_16k.wav")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-f", "s16le", "-ar", "48000", "-ac", "2",
        "-i", src_path,
        "-ar", "16000", "-ac", "1",
        dst_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    os.unlink(src_path)
    return dst_path

# -------------------------------------------------
# Incremental (chunked) transcription
# -------------------------------------------------

async def transcribe_pending_chunk(guild_id: int, sink: ChunkedWaveSink, session: VoiceSession) -> None:
    """Drain the sink's pending buffer and transcribe each user's audio in parallel."""
    pending = sink.drain_pending()
    if not pending:
        return

    async def process_user(user_id: int, raw: bytes) -> None:
        if not raw:
            return
        logger.info("Transcribing chunk for user %d: %d bytes of PCM", user_id, len(raw))
        try:
            wav_16k_path = await convert_pcm_to_16k_mono(raw)
            text = await transcribe_wav(wav_16k_path)
            os.unlink(wav_16k_path)
        except Exception as e:
            logger.exception("Chunk transcription failed for user %d: %s", user_id, e)
            return
        if text:
            session.transcript_chunks.setdefault(user_id, []).append(text)
            logger.info("Chunk transcribed for user %d: %d chars", user_id, len(text))
        else:
            logger.info("Chunk for user %d produced no text (silence?)", user_id)

    await asyncio.gather(*(process_user(uid, raw) for uid, raw in pending.items()))


async def periodic_transcription(
    guild_id: int, sink: ChunkedWaveSink, stop_event: asyncio.Event
) -> None:
    """Background task: every CHUNK_INTERVAL seconds, drain and transcribe pending audio.
    When stop_event is set, do one final pass and exit."""
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=CHUNK_INTERVAL)
            # stop_event fired — do final transcription then exit
            session = voice_sessions.get(guild_id)
            if session:
                await transcribe_pending_chunk(guild_id, sink, session)
            return
        except asyncio.TimeoutError:
            pass  # normal interval elapsed

        session = voice_sessions.get(guild_id)
        if not session or not session.active:
            return
        await transcribe_pending_chunk(guild_id, sink, session)

# -------------------------------------------------
# Ollama summary
# -------------------------------------------------

async def generate_summary(transcript_text: str, session_info: str) -> str:
    prompt = (
        "You are a dungeon master's scribe. Summarize ONLY what is explicitly said in the transcript below. "
        "Do NOT invent, infer, or add anything that is not directly stated. "
        "Do NOT fill in missing details or imagine what might have happened. "
        "If the transcript contains very little content, write a brief, honest summary of only what was said — even if that is just a sentence or two. "
        "Only include a section if the transcript actually contains relevant content for it:\n"
        "- Key events or plot points mentioned\n"
        "- Decisions or plans the players discussed\n"
        "- Notable moments or role-play that occurred\n"
        "- Unresolved questions raised in the conversation\n\n"
        f"Session info: {session_info}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Summary (based strictly on the transcript above):"
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
        f"**Started:** {started_at.strftime('%Y-%m-%d %H:%M %Z')}  \n"
        f"**Ended:** {ended_at.strftime('%Y-%m-%d %H:%M %Z')}  \n"
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

    rc = await run("git", "pull", "--rebase")
    if rc != 0:
        logger.error("git pull --rebase failed for '%s' — aborting rebase, skipping push.", filename)
        await run("git", "rebase", "--abort")
        return
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

async def handle_recording_finished(guild_id: int, sink: ChunkedWaveSink) -> None:
    ended_at = datetime.now().astimezone()
    session = voice_sessions.get(guild_id)
    if not session:
        return

    # Signal the chunk task to do one final drain+transcribe, then wait for it.
    session.stop_event.set()
    if session.chunk_task and not session.chunk_task.done():
        try:
            await asyncio.wait_for(session.chunk_task, timeout=600)
        except asyncio.TimeoutError:
            logger.warning("Final chunk transcription timed out — transcript may be incomplete.")

    guild = bot.get_guild(guild_id)
    text_channel = guild.get_channel(session.text_channel_id) if guild else None
    if text_channel is None:
        return

    await text_channel.send("Recording ended. Building transcript...")

    transcript_lines: List[str] = []
    participants: List[str] = []

    # audio_data keys cover every user who spoke, even if their last chunk is already in transcript_chunks
    all_user_ids = set(session.transcript_chunks.keys()) | set(sink.audio_data.keys())
    for user_id in all_user_ids:
        member = guild.get_member(user_id) if guild else None
        discord_name = member.display_name if member else f"User {user_id}"
        char_name = session_manager.get_character(guild_id, user_id)
        speaker_label = f"{char_name} ({discord_name})" if char_name else discord_name
        participants.append(speaker_label)

        chunks = session.transcript_chunks.get(user_id, [])
        text = " ".join(chunks) if chunks else "[no audio]"
        transcript_lines.append(f"**{speaker_label}**: {text}")

    full_transcript = "\n\n".join(transcript_lines)
    await post_chunked(text_channel, "**Transcript:**\n\n" + full_transcript)

    vc_name = session.voice_channel_name or "unknown"

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


# -------------------------------------------------
# Slash commands
# -------------------------------------------------

@bot.slash_command(name="ping", description="Check if the bot is alive.", guild_ids=GUILD_IDS or None)
async def ping(ctx: discord.ApplicationContext) -> None:
    await ctx.respond("Pong.", ephemeral=True)


@bot.slash_command(
    name="tt_start",
    description="Start live transcription in your current voice channel.",
    guild_ids=GUILD_IDS or None,
)
async def start_command(ctx: discord.ApplicationContext) -> None:
    guild = ctx.guild
    user = ctx.author

    if guild is None or not isinstance(user, discord.Member):
        await ctx.respond("This command can only be used in a server.", ephemeral=True)
        return

    if not user.voice or not user.voice.channel:
        await ctx.respond("You must be in a voice channel to start transcription.", ephemeral=True)
        return

    existing = voice_sessions.get(guild.id)
    if existing and existing.active and existing.voice_client.is_connected():
        await ctx.respond("Transcription is already active in this server.", ephemeral=True)
        return

    voice_channel = user.voice.channel
    text_channel = ctx.channel

    if not isinstance(text_channel, discord.TextChannel):
        await ctx.respond("This command must be used in a text channel.", ephemeral=True)
        return

    await ctx.defer()

    try:
        voice_client = await voice_channel.connect()
    except discord.ClientException:
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client is None:
            await ctx.followup.send("Failed to connect to the voice channel.")
            return

    sink = ChunkedWaveSink()

    session = VoiceSession(
        guild_id=guild.id,
        voice_client=voice_client,
        text_channel_id=text_channel.id,
        started_by_id=user.id,
    )

    def finished_callback(sink: discord.sinks.Sink, *args) -> None:
        bot.loop.create_task(handle_recording_finished(guild.id, sink))

    voice_client.start_recording(sink, finished_callback, None)

    session.chunk_task = bot.loop.create_task(
        periodic_transcription(guild.id, sink, session.stop_event)
    )
    voice_sessions.add(session)

    await ctx.followup.send(
        f"Started transcription in {voice_channel.mention}. "
        "Use `/tt_stop` to end and post the transcript."
    )


@bot.slash_command(
    name="tt_stop",
    description="Stop the current transcription and post the transcript.",
    guild_ids=GUILD_IDS or None,
)
async def stop_command(ctx: discord.ApplicationContext) -> None:
    guild = ctx.guild
    if guild is None:
        await ctx.respond("This command can only be used in a server.", ephemeral=True)
        return

    session = voice_sessions.get(guild.id)
    if not session or not session.active:
        await ctx.respond("No active transcription in this server.", ephemeral=True)
        return

    await ctx.defer()

    vc = session.voice_client
    if vc and vc.is_connected():
        session.voice_channel_name = vc.channel.name if vc.channel else "unknown"
        vc.stop_recording()  # triggers finished_callback
        await vc.disconnect()

    voice_sessions.stop(guild.id)

    await ctx.followup.send(
        "Stopping transcription. I will post the transcript here when it is ready."
    )


@bot.slash_command(
    name="status",
    description="Show the status of the current transcription session.",
    guild_ids=GUILD_IDS or None,
)
async def status_command(ctx: discord.ApplicationContext) -> None:
    guild = ctx.guild
    if guild is None:
        await ctx.respond("This command can only be used in a server.", ephemeral=True)
        return

    session = voice_sessions.get(guild.id)
    if not session or not session.active:
        await ctx.respond("No active transcription session in this server.", ephemeral=True)
        return

    vc = session.voice_client
    vc_name = vc.channel.name if vc and vc.channel else "unknown"
    started_at_str = session.started_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    msg = (
        f"Transcription is active.\n"
        f"Voice channel: **{vc_name}**\n"
        f"Text channel: <#{session.text_channel_id}>\n"
        f"Started by: <@{session.started_by_id}>\n"
        f"Started at: {started_at_str}"
    )
    await ctx.respond(msg, ephemeral=True)


@bot.slash_command(
    name="set_character",
    description="Associate a Discord user with a character name for transcripts.",
    guild_ids=GUILD_IDS or None,
)
async def set_character_command(
    ctx: discord.ApplicationContext,
    character_name: discord.Option(str, "The character name to associate with this user."),
    user: discord.Option(discord.Member, "The Discord user to set the character for (defaults to yourself).", required=False) = None,
) -> None:
    guild = ctx.guild
    if guild is None:
        await ctx.respond("This command can only be used in a server.", ephemeral=True)
        return

    target = user or ctx.author
    session_manager.set_character(guild.id, target.id, character_name)

    if target.id == ctx.author.id:
        msg = f"Set your character name to: **{character_name}**"
    else:
        msg = f"Set character name for {target.mention} to: **{character_name}**"

    await ctx.respond(msg, ephemeral=True)

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
        await bot.sync_commands(guild_ids=[guild.id])
        await ctx.send("Synced commands to this guild.")
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
