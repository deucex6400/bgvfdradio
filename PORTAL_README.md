# SDR Bot Portal + CTCSS tone squelch + Login

Features:
- Session login (username **admin**, password **3400**) — override with PORTAL_USER/PORTAL_PASS; set PORTAL_SECRET.
- Per-channel defaults in `channels.json`: `mhz`, optional `ctcss_hz`, `ctcss_notch`.
- Tone-gated audio: bot mutes frames unless configured CTCSS tone is present; optional notch via FFmpeg.
- Web portal to tune, edit channels (including tones), change settings, view health.

## Run (Pi)
```bash
python -m pip install fastapi uvicorn jinja2 discord.py[voice] jinja2 starlette
sudo apt-get install -y rtl-sdr ffmpeg libopus0
export DISCORD_CFG="{'token':'<BOT_TOKEN>','guildId':<GUILD_ID>,'voiceChannelId':<VOICE_ID>,'prefix':'!'}"
export PORTAL_USER='admin'
export PORTAL_PASS='3400'
export PORTAL_SECRET='change-this-in-prod-please'
python run_portal.py
```
Open http://<pi>:8080 — login required.

### channels.json format
```json
{
  "Nav Fire": { "mhz": 154.1075, "ctcss_hz": 100.0, "ctcss_notch": true },
  "Nav Med": 154.235,
  "VFire22": { "mhz": 154.265, "ctcss_hz": 192.8 },
  "NOAA6":   { "mhz": 162.525 }
}
```
