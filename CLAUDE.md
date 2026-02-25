# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Bot

```bash
cd transcript_bot
pipenv install
pipenv run python transcript_bot.py
```

The bot requires Python 3.10 exactly (see `Pipfile`). Using a different Python version will break the environment.

## External Dependencies

These must be set up before the bot works:
- **ffmpeg** on `PATH` — used to resample Discord's 48 kHz stereo PCM to 16 kHz mono WAV before Whisper
- **Ollama** running locally with the configured model pulled: `ollama pull llama3.2`
- **libopus** — path set via `OPUS_LIBRARY` in `.env` (e.g. `/lib/x86_64-linux-gnu/libopus.so.0`)
- **TRANSCRIPT_DIR** should point to a git repo if auto-push is desired (e.g. an Obsidian vault)

## Configuration

All config lives in `.env` at the repo root. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `DISCORD_TOKEN` | required | Bot token |
| `GUILD_ID` | — | Comma-separated guild IDs; empty = global (slow) |
| `OPUS_LIBRARY` | — | Absolute path to libopus.so |
| `SUMMARY_CHANNEL_ID` | — | Channel for LLM summaries |
| `TRANSCRIPT_DIR` | `./transcripts` | Obsidian vault path |
| `CAMPAIGN_NAME` | `TavernTalk` | Used in filenames and Markdown |
| `WHISPER_MODEL` | `small` | tiny/base/small/medium |
| `OLLAMA_MODEL` | `llama3.2` | Must be pulled |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `CHUNK_INTERVAL` | `60` | Seconds between mid-session transcription passes |

## Architecture

Everything is in a single file: `transcript_bot/transcript_bot.py`.

**Audio pipeline (recording → transcript):**
1. `ChunkedWaveSink` (extends `discord.sinks.WaveSink`) — captures raw Discord PCM. It maintains both the standard `audio_data` dict (discord.py-managed, used at session end) and a parallel thread-safe `_pending` dict that `drain_pending()` clears atomically.
2. `periodic_transcription` — background `asyncio.Task` that wakes every `CHUNK_INTERVAL` seconds, calls `drain_pending()`, resamples each user's PCM via `ffmpeg`, and sends to Whisper. Results accumulate in `VoiceSession.transcript_chunks`.
3. On `/tt_stop`: `stop_event` is set, waking the task for a final drain pass. `handle_recording_finished` awaits the task, then assembles the full transcript from `transcript_chunks`.

**State management:**
- `VoiceSessionManager` (`voice_sessions`) — holds active `VoiceSession` objects per guild (recording state, chunk task, stop event)
- `SessionManager` (`session_manager`) — holds character name mappings per guild, loaded lazily from `data/characters/{guild_id}.json`, saved on every `/set_character`

**Post-recording flow in `handle_recording_finished`:**
1. Posts raw transcript to the text channel (chunked to stay under Discord's 2000-char limit via `post_chunked`)
2. Posts a placeholder to `SUMMARY_CHANNEL_ID`
3. Calls Ollama for LLM summary (can take minutes on CPU)
4. Saves Obsidian Markdown file with YAML frontmatter to `TRANSCRIPT_DIR`
5. Runs `git pull --rebase && git add && git commit && git push` on `TRANSCRIPT_DIR`
6. Edits the placeholder with the final summary

**Slash commands:** `/ping`, `/tt_start`, `/tt_stop`, `/status`, `/set_character`
**Prefix command (admin only):** `!sync_guild` — manually re-sync slash commands to a guild

## Character Persistence

Character maps are stored in `data/characters/{guild_id}.json` as `{"user_id": "character_name"}`. These are committed to the repo (not gitignored) so they persist across bot restarts.

## Discord Library

Uses **py-cord** (not discord.py). The slash command API, `discord.ApplicationContext`, `discord.Option`, and `bot.slash_command` decorator are py-cord specific. Do not confuse with discord.py's app command extension.
