#!/usr/bin/env python3
"""
TavernTalk bot — Pycord 2.4.1

Features:
- Loads .env file automatically (for DISCORD_TOKEN, etc.)
- Clean logging
- Robust /join and /leave slash commands with fast ACK
- Safe reply helper
- Voice handshake timeouts & detailed logs
"""

import os, sys, asyncio, logging, shutil, discord
from discord.ext import commands
from dotenv import load_dotenv

# ---------- Load environment ----------
load_dotenv()

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("taverntalk")
logging.getLogger("discord").setLevel(logging.INFO)

# Optional: quiet PyNaCl warning
discord.VoiceClient.warn_nacl = False

# ---------- Intents & Bot ----------
intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
bot = commands.Bot(intents=intents)

# ---------- Utilities ----------
def print_banner():
    opus_ok = discord.opus.is_loaded()
    print(f"opus loaded: {opus_ok}")
    print("discord lib: pycord 2.4.1")

def require_token() -> str:
    token = os.getenv("DISCORD_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        print("ERROR: Set DISCORD_TOKEN (or BOT_TOKEN) in your .env or environment.", file=sys.stderr)
        sys.exit(1)
    return token

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

async def safe_reply(ctx: discord.ApplicationContext, content: str, *, ephemeral: bool = True):
    """Reply once; if already deferred/responded, use followup."""
    try:
        if not getattr(ctx, "responded", False) and not getattr(ctx, "deferred", False):
            await ctx.respond(content, ephemeral=ephemeral)
        else:
            await ctx.followup.send(content, ephemeral=ephemeral)
    except discord.HTTPException:
        log.exception("Reply failed")

async def safe_defer(ctx, ephemeral: bool = True):
    # Only defer if we haven't already responded/deferred
    if not getattr(ctx, "responded", False) and not getattr(ctx, "deferred", False):
        await ctx.defer(ephemeral=ephemeral)

# ---------- Events ----------
@bot.event
async def on_ready():
    try:
        await bot.sync_commands()
    except Exception as e:
        log.warning("Command sync warning: %s", e)

    user = bot.user
    if user:
        print(f"Logged in as {user} (ID: {user.id})")
    else:
        print("Logged in (no user?)")

    if not have_ffmpeg():
        log.warning("ffmpeg not found on PATH. Install it if you plan to record or process audio.")

@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    if member.id == bot.user.id:
        b = before.channel.name if before and before.channel else None
        a = after.channel.name if after and after.channel else None
        log.info("Voice state (self): %s -> %s", b, a)

# ---------- Commands ----------
@bot.slash_command(description="Join your voice channel (or a specified one).")
async def join(
    ctx: discord.ApplicationContext,
    channel: discord.Option(discord.VoiceChannel, description="Voice channel to join", required=False) = None
):
    # Defer quickly to avoid "application did not respond"
    if not getattr(ctx, "responded", False) and not getattr(ctx, "deferred", False):
        await ctx.defer(ephemeral=True)

    target = channel or (ctx.author.voice and ctx.author.voice.channel)
    if not target:
        return await ctx.followup.send("Join a voice channel first, or pass one to `/join`.", ephemeral=True)

    vc = ctx.guild.voice_client
    try:
        if vc and vc.channel and vc.channel.id == target.id:
            return await ctx.followup.send(f"Already in **{target.name}**.", ephemeral=True)

        if vc and vc.channel and vc.channel.id != target.id:
            await vc.move_to(target)
            return await ctx.followup.send(f"Moved to **{target.name}**.", ephemeral=True)

        # Pycord 2.4.1: no self_deaf/self_mute kwarg here
        await asyncio.wait_for(target.connect(reconnect=True), timeout=12)
        await ctx.followup.send(f"Connected to **{target.name}**.", ephemeral=True)

    except asyncio.TimeoutError:
        await ctx.followup.send("Timed out during voice handshake (check outbound UDP).", ephemeral=True)
    except Exception as e:
        await ctx.followup.send(f"Error: `{type(e).__name__}: {e}`", ephemeral=True)

@bot.slash_command(description="Leave the current voice channel.")
async def leave(ctx: discord.ApplicationContext):
    await ctx.defer(ephemeral=True)  # <-- changed
    vc = ctx.guild.voice_client
    if not vc:
        return await ctx.followup.send("I’m not in a voice channel.", ephemeral=True)
    try:
        await vc.disconnect(force=True)
        await ctx.followup.send("Disconnected.", ephemeral=True)
    except Exception as e:
        await ctx.followup.send(f"Error disconnecting: `{type(e).__name__}: {e}`", ephemeral=True)

import pathlib
pathlib.Path("recordings").mkdir(exist_ok=True)

@bot.slash_command(description="Start recording the current voice channel to WAV files.")
async def record_start(ctx: discord.ApplicationContext):
    await ctx.defer(ephermeral=True)
    vc = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        return await ctx.followup.send("I’m not in a voice channel. Use `/join` first.", ephemeral=True)

    if getattr(vc, "_is_recording", False):
        return await ctx.followup.send("Already recording.", ephemeral=True)

    sink = discord.sinks.WaveSink()  # 48kHz mono per-user WAVs

    def finished(sink: discord.sinks.Sink, *args):
        # One file per user
        for user_id, data in sink.audio_data.items():
            uname = f"{data.user.name}_{data.user.id}".replace("#", "_")
            out = pathlib.Path("recordings") / f"{uname}.wav"
            with open(out, "wb") as f:
                f.write(data.file.getbuffer())
        # mark not recording
        setattr(vc, "_is_recording", False)

    vc.start_recording(sink, finished, ctx)
    setattr(vc, "_is_recording", True)
    await ctx.followup.send("Recording started. Use `/record_stop` to finish.", ephemeral=True)

@bot.slash_command(description="Stop an active recording and save files.")
async def record_stop(ctx: discord.ApplicationContext):
    await ctx.defer(ephermeral=True)
    vc = ctx.guild.voice_client
    if not vc or not getattr(vc, "_is_recording", False):
        return await ctx.followup.send("Not currently recording.", ephemeral=True)
    try:
        vc.stop_recording()  # triggers the finished() callback above
        await ctx.followup.send("Recording stopped. Files saved in `recordings/`.", ephemeral=True)
    except Exception as e:
        await ctx.followup.send(f"Stop failed: `{type(e).__name__}: {e}`", ephemeral=True)


# ---------- Main ----------
def main():
    print_banner()
    token = require_token()

    # Force-load Opus by absolute path if auto-detect fails
    if not discord.opus.is_loaded():
        for cand in (
            "/lib/x86_64-linux-gnu/libopus.so",
            "/lib/x86_64-linux-gnu/libopus.so.0",
            "libopus.so.0",   # some systems resolve this
            "opus",           # fallback to generic name
        ):
            try:
                discord.opus.load_opus(cand)
                break
            except Exception:
                pass
    print(f"opus loaded: {discord.opus.is_loaded()}")

    bot.run(token, reconnect=True)

if __name__ == "__main__":
    main()

