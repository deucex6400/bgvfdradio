# -*- coding: utf-8 -*-
"""
SDR Discord Bot (Windows & Linux) — VHF/UHF channels via rtl_fm -> FFmpeg -> Discord voice

- Auto-detects OS and uses correct executable names (Windows .exe vs Linux binaries).
- Loads channel list from channels.json (override via CHANNELS_JSON env var).
- Provides both slash and prefix commands to list/reload/tune channels.
- Uses rtl_fm for USB demod and FFmpeg for resample/filters; always outputs stereo
  20 ms @ 48 kHz frames to Discord to avoid pitch/timebase issues on Linux.
- Defaults applied from user-provided settings.

Environment (preferred single-quoted JSON style):
    export DISCORD_CFG="{'token':'<TOKEN>','guildId':123,'voiceChannelId':456,'prefix':'!'}"
    # Optional overrides
    export RTL_FM_EXE=rtl_fm
    export FFMPEG_EXE=ffmpeg
    export CHANNELS_JSON=channels.json

Requirements:
    - Windows: rtl_fm.exe, ffmpeg.exe on PATH; RTL-SDR driver via Zadig.
    - Linux/Pi: sudo apt install rtl-sdr ffmpeg ; close other SDR apps.

"""
import os
import sys
import asyncio
import signal
import subprocess
import shutil
from dataclasses import dataclass, asdict
import json
import discord
from discord import app_commands
from discord.ext import commands

# -------------------- Config parsing --------------------
def parse_cfg():
    cfg = {}
    raw = os.getenv('DISCORD_CFG', '')
    if raw:
        try:
            j = raw.strip()
            if (j.startswith('"') and j.endswith('"')) or (j.startswith("'") and j.endswith("'")):
                j = j[1:-1]
            j2 = j.replace("'", '"')
            cfg = json.loads(j2)
        except Exception:
            cfg = {}
    token = os.getenv('DISCORD_TOKEN', cfg.get('token', ''))
    gid = os.getenv('GUILD_ID', str(cfg.get('guildId', '')))
    vid = os.getenv('VOICE_CHANNEL_ID', str(cfg.get('voiceChannelId', '')))
    pfx = os.getenv('COMMAND_PREFIX', cfg.get('prefix', '!'))
    out = {'token': token, 'prefix': pfx}
    try:
        if gid:
            out['guildId'] = int(gid)
    except Exception:
        pass
    try:
        if vid:
            out['voiceChannelId'] = int(vid)
    except Exception:
        pass
    return out

CFG = parse_cfg()

# -------------------- Executables / OS detection --------------------
DEFAULT_FFMPEG = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
DEFAULT_RTLFM  = 'rtl_fm.exe' if os.name == 'nt' else 'rtl_fm'
FFMPEG_EXE = os.getenv('FFMPEG_EXE', DEFAULT_FFMPEG)
RTL_FM_EXE = os.getenv('RTL_FM_EXE', DEFAULT_RTLFM)
CHANNELS_JSON = os.getenv('CHANNELS_JSON', 'channels.json')
RTL_DEVICE_INDEX = os.getenv('RTL_DEVICE_INDEX')  # optional: '0','1',...

# Windows process flag to hide extra consoles
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

# -------------------- Channel loading --------------------
def load_channels(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        out = {}
        for name, mhz in data.items():
            try:
                out[name] = int(round(float(mhz) * 1_000_000))
            except Exception:
                pass
        return out
    except Exception:
        return {}

CHANNELS = load_channels(CHANNELS_JSON)

# -------------------- SDR settings --------------------
@dataclass
class SDRSettings:
    gain_db: int = 35
    squelch: int = 1
    samplerate: int = 12000
    oversampling: int = 4
    ppm: int = 0
    atan_mode: str = 'fast'   # std|fast|lut
    deemp: bool = True
    offset_tuning: bool = False
    dc_block: bool = False
    fir_size: int = 0         # user requested default = 0
    center_offset_hz: int = 0
    output_stereo: bool = True
    output_clean: bool = True

SETTINGS = SDRSettings()

# 20 ms @ 48 kHz = 960 samples per channel
SAMPLES_PER_20MS = 960

# -------------------- Audio pipeline (USB-only rtl_fm) --------------------
class NOAAStreamSource(discord.AudioSource):
    def __init__(self, freq_hz: int, settings: SDRSettings):
        self.base_freq_hz = freq_hz
        self.settings = settings
        self.rf_proc = None
        self.ff_proc = None
        self._start_pipeline()

    @property
    def freq_hz(self) -> int:
        return self.base_freq_hz + int(self.settings.center_offset_hz or 0)

    def _start_pipeline(self):
        # rtl_fm command
        rf_cmd = [
            RTL_FM_EXE,
            '-f', str(self.freq_hz),
            '-M', 'fm',
            '-s', str(self.settings.samplerate),
            '-g', str(self.settings.gain_db),
            '-l', str(self.settings.squelch),
            '-o', str(self.settings.oversampling),
            '-A', self.settings.atan_mode,
        ]
        # optional quality flags
        if self.settings.deemp:
            rf_cmd += ['-E', 'deemp']
        if self.settings.offset_tuning:
            rf_cmd += ['-E', 'offset']
        if self.settings.dc_block:
            rf_cmd += ['-E', 'dc']
        if self.settings.fir_size in (0, 9):
            rf_cmd += ['-F', str(self.settings.fir_size)]
        if self.settings.ppm:
            rf_cmd += ['-p', str(self.settings.ppm)]
        if RTL_DEVICE_INDEX not in (None, ''):
            rf_cmd += ['-d', str(RTL_DEVICE_INDEX)]
        # raw PCM s16le to stdout
        rf_cmd += ['-']

        # FFmpeg filters
        stereo_pan = 'pan=stereo|c0=c0|c1=c0'  # Linux-safe syntax
        clean_chain = 'highpass=f=300,lowpass=f=4000,afftdn=nr=15:nf=-30:tn=1'
        af_chain = stereo_pan if not self.settings.output_clean else f'{stereo_pan},{clean_chain}'
        ff_cmd = [
            FFMPEG_EXE, '-hide_banner', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(self.settings.samplerate), '-ac', '1', '-i', 'pipe:0',
            '-af', af_chain,
            '-ac', '2', '-ar', '48000', '-f', 's16le', 'pipe:1'
        ]

        self.rf_proc = subprocess.Popen(
            rf_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW
        )
        self.ff_proc = subprocess.Popen(
            ff_cmd, stdin=self.rf_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW
        )

    def _frame_len_bytes(self) -> int:
        # Always deliver stereo frames: 20 ms @ 48 kHz, s16le
        return SAMPLES_PER_20MS * 2 * 2  # 960 * 2 bytes * 2 channels = 3840

    def read(self):
        try:
            need = self._frame_len_bytes()
            return self.ff_proc.stdout.read(need)
        except Exception:
            return b''

    def cleanup(self):
        for p in (self.ff_proc, self.rf_proc):
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass

    def is_opus(self):
        return False

# -------------------- Discord bot setup --------------------
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix=CFG.get('prefix', '!'), intents=intents)
bot.remove_command('help')

voice_client: discord.VoiceClient | None = None
current_source: NOAAStreamSource | None = None
current_freq_hz: int | None = None

# -------------------- Helpers --------------------
def on_player_after(error):
    if error:
        print(f"[ERROR] Player error: {error}")

async def connect_voice(channel_id: int | None, ctx: commands.Context | None = None):
    global voice_client
    chan_obj = None
    if channel_id:
        chan = await bot.fetch_channel(channel_id)
        if isinstance(chan, discord.VoiceChannel):
            chan_obj = chan
        else:
            raise RuntimeError('VOICE_CHANNEL_ID is not a voice channel')
    elif ctx and getattr(ctx.author, 'voice', None) and ctx.author.voice and ctx.author.voice.channel:
        chan_obj = ctx.author.voice.channel
    else:
        raise RuntimeError('No voice channel provided and author not in a voice channel.')

    if voice_client and voice_client.is_connected():
        await voice_client.move_to(chan_obj)
    else:
        voice_client = await chan_obj.connect()

async def _restart_pipeline():
    global voice_client, current_source, current_freq_hz
    if not voice_client or not voice_client.is_connected():
        raise RuntimeError('Not connected to a voice channel')
    if current_freq_hz is None:
        raise RuntimeError('No frequency set; use tune command')
    if voice_client.is_playing():
        voice_client.stop()
    if current_source:
        current_source.cleanup()
    current_source = NOAAStreamSource(freq_hz=current_freq_hz, settings=SETTINGS)
    voice_client.play(current_source, after=on_player_after)

async def play_noaa(freq_hz: int):
    global current_freq_hz
    current_freq_hz = freq_hz
    await _restart_pipeline()

# -------------------- Health checks --------------------
def _cmd_version(cmd, args=['-version'], timeout=2):
    try:
        proc = subprocess.run([cmd] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        if proc.returncode == 0:
            first = proc.stdout.decode(errors='ignore').splitlines()[:2]
            return True, '\n'.join(first)
        else:
            return False, f'exit {proc.returncode}'
    except FileNotFoundError:
        return False, 'not found'
    except Exception as e:
        return False, str(e)

def _check_ffmpeg():
    ff = shutil.which(FFMPEG_EXE) or FFMPEG_EXE
    ok, msg = _cmd_version(ff)
    return {'present': ok, 'info': msg, 'path': ff}

def _check_rtl_fm():
    rf = shutil.which(RTL_FM_EXE) or RTL_FM_EXE
    if not shutil.which(RTL_FM_EXE):
        exists = os.path.isfile(rf)
        if not exists:
            return {'present': False, 'info': 'not found', 'path': rf}
    ok, msg = _cmd_version(rf, ['-h'])
    return {'present': True, 'info': f'path={rf}\n{msg}'}

def _check_pynacl_opus():
    info = {}
    info['discord_py_version'] = getattr(discord, '__version__', 'unknown')
    try:
        import nacl  # noqa: F401
        info['pynacl'] = True
    except Exception:
        info['pynacl'] = False
    if discord.opus.is_loaded():
        info['opus_loaded'] = True
    else:
        tried = []
        for lib in ('opus.dll', 'libopus.so.0', 'libopus.so'):
            try:
                discord.opus.load_opus(lib)
                info['opus_loaded'] = True
                info['opus_lib'] = lib
                break
            except Exception as e:
                tried.append(f'{lib}: {e.__class__.__name__}')
        if not info.get('opus_loaded'):
            info['opus_loaded'] = False
            info['opus_tried'] = tried
    return info

def _check_intents():
    return {
        'message_content': bool(bot.intents.message_content),
        'voice_states': bool(bot.intents.voice_states),
    }

async def _check_perms_and_status():
    summary = {}
    vc_id = CFG.get('voiceChannelId')
    if vc_id:
        try:
            chan = await bot.fetch_channel(vc_id)
            if isinstance(chan, discord.VoiceChannel):
                guild = chan.guild
                me = guild.me
                perms = chan.permissions_for(me)
                summary.update({
                    'channel_id': chan.id,
                    'channel_name': chan.name,
                    'guild_id': guild.id,
                    'guild_name': guild.name,
                    'perm_connect': perms.connect,
                    'perm_speak': perms.speak,
                })
            else:
                summary['error'] = 'Configured VOICE_CHANNEL_ID is not a voice channel.'
        except Exception as e:
            summary['error'] = f'Cannot fetch VOICE_CHANNEL_ID: {e.__class__.__name__}'
    else:
        summary['note'] = 'VOICE_CHANNEL_ID not set; use !join while you are in a voice channel.'
    global voice_client, current_source, current_freq_hz
    summary['connected'] = bool(voice_client and voice_client.is_connected())
    summary['playing'] = bool(voice_client and voice_client.is_playing())
    summary['latency'] = (round(getattr(voice_client, 'average_latency', 0.0), 4) if voice_client else None)
    summary['channel_active'] = getattr(getattr(voice_client, 'channel', None), 'name', None)
    summary['freq_mhz'] = (round(current_freq_hz / 1e6, 6) if current_freq_hz else None)
    summary['settings'] = asdict(SETTINGS)
    summary['backend'] = 'rtl_fm (USB)'
    summary['device_index'] = RTL_DEVICE_INDEX
    summary['channels_loaded'] = len(CHANNELS)
    return summary

# -------------------- Help text --------------------
def build_prefix_help(prefix: str) -> str:
    txt = []
    txt.append('**SDR Bot - Prefix Commands**')
    txt.append(f'- {prefix}start : Join configured voice channel. If channels.json has entries, auto-tunes first channel.')
    txt.append(f'- {prefix}join : Join voice channel (configured or your current one).')
    txt.append(f'- {prefix}leave : Stop stream and disconnect.')
    txt.append(f'- {prefix}chan_tune <Name|MHz> : Tune a channel from channels.json (e.g., "Nav Fire") or MHz.')
    txt.append(f'- {prefix}chan_list : List all channels from channels.json.')
    txt.append(f'- {prefix}chan_reload : Reload channels.json without restarting the bot.')
    txt.append(f'- {prefix}wx_gain <dB> : Set tuner gain.')
    txt.append(f'- {prefix}wx_squelch <level> : Squelch (0 disables).')
    txt.append(f'- {prefix}wx_bw <Hz> : Demod sample rate (rtl_fm -s).')
    txt.append(f'- {prefix}wx_oversampling <n> : Oversampling (rtl_fm -o).')
    txt.append(f'- {prefix}wx_ppm <ppm> : PPM correction.')
    txt.append(f'- {prefix}wx_deemp on|off : De-emphasis filter.')
    txt.append(f'- {prefix}wx_offset on|off : Offset tuning.')
    txt.append(f'- {prefix}wx_dc on|off : DC blocking.')
    txt.append(f'- {prefix}wx_fir 0|9 : FIR downsample size.')
    txt.append(f'- {prefix}wx_centeroffset <kHz> : Center offset +/- in kHz.')
    txt.append(f'- {prefix}wx_atan std|fast|lut : atan math.')
    txt.append(f'- {prefix}wx_defaults : Apply baseline settings.')
    txt.append(f'- {prefix}wx_stereo on|off : Toggle stereo upmix (output stays stereo to Discord).')
    txt.append(f'- {prefix}wx_clean on|off : Toggle cleanup (HP/LP/denoise).')
    txt.append(f'- {prefix}health : Health report.')
    return '\n'.join(txt)

def build_slash_help() -> str:
    txt = []
    txt.append('**SDR Bot - Slash Commands**')
    txt.append('- /chan_tune name|mhz')
    txt.append('- /chan_list')
    txt.append('- /chan_reload')
    txt.append('- /wx_set gain squelch bandwidth oversampling ppm deemp offset center_offset_khz atan')
    txt.append('- /wx_gain db')
    txt.append('- /wx_defaults')
    txt.append('- /wx_stereo on|off')
    txt.append('- /wx_clean on|off')
    txt.append('- /wx_dc on|off')
    txt.append('- /wx_fir size(0|9)')
    txt.append('- /wx_stop')
    txt.append('- /health')
    txt.append('- /help')
    return '\n'.join(txt)

# -------------------- Events --------------------
@bot.event
async def on_ready():
    global current_channel_name

    print(f"[INFO] Logged in as {bot.user} (id={bot.user.id})")
    gid = CFG.get('guildId')
    if gid:
        await bot.tree.sync(guild=discord.Object(id=gid))
        print(f"[INFO] Slash commands synced to guild {gid}")
    else:
        await bot.tree.sync()
        print('[INFO] Global slash commands synced (may take a minute)')
    vc_id = CFG.get('voiceChannelId')
    if vc_id:
        await connect_voice(vc_id)
        # Auto-tune first channel if available
        if CHANNELS:
            first_name = sorted(CHANNELS.keys())[0]
            await play_noaa(CHANNELS[first_name])
            print(f"[INFO] Auto-tuned to {first_name} ({CHANNELS[first_name]/1e6:.6f} MHz)")
    print(f"[INFO] Prefix commands active with prefix '{CFG.get('prefix','!')}'")

# -------------------- Slash commands --------------------
@bot.tree.command(name='chan_tune', description='Tune a channel from channels.json or a custom frequency (MHz)')
@app_commands.describe(name='Channel name (e.g., Nav Fire)', mhz='Custom frequency in MHz, e.g., 154.265')
async def chan_tune(interaction: discord.Interaction, name: str = None, mhz: float = None):
    global current_channel_name

    try:
        if name:
            if name not in CHANNELS:
                await interaction.response.send_message(f"Unknown channel '{name}'. Try /chan_list.", ephemeral=True)
                return
            await play_noaa(CHANNELS[name])
    global current_channel_meta
    current_channel_meta = CHANNEL_META.get(name, {})
            await interaction.response.send_message(f"Tuned to {name} ({CHANNELS[name]/1e6:.6f} MHz)")
        elif mhz:
            await play_noaa(int(mhz * 1e6))
    global current_channel_meta
    current_channel_meta = None
            await interaction.response.send_message(f"Tuned to {mhz:.6f} MHz")
        else:
            await interaction.response.send_message('Provide a channel name or MHz.', ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='chan_list', description='List all channels from channels.json')
async def chan_list(interaction: discord.Interaction):
    try:
        if not CHANNELS:
            await interaction.response.send_message('No channels loaded. Use /chan_reload after placing channels.json.', ephemeral=True)
            return
        lines = [f"- {k}: {v/1e6:.6f} MHz" for k, v in sorted(CHANNELS.items())]
        await interaction.response.send_message('\n'.join(lines), ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='chan_reload', description='Reload channels.json into memory')
async def chan_reload(interaction: discord.Interaction):
    global CHANNELS
    try:
        CHANNELS = load_channels(CHANNELS_JSON)
        await interaction.response.send_message(f'Reloaded {len(CHANNELS)} channels from {CHANNELS_JSON}.', ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

# Existing wx_set style controls retained for tuning params
@bot.tree.command(name='wx_set', description='Set rtl_fm parameters.')
@app_commands.describe(
    gain='tuner gain dB',
    squelch='squelch level; 0 disables',
    bandwidth='demod sample rate in Hz (rtl_fm -s)',
    oversampling='oversampling (rtl_fm -o)',
    ppm='PPM correction (rtl_fm -p)',
    deemp='de-emphasis (on/off)',
    offset='offset tuning (on/off)',
    center_offset_khz='center frequency offset in kHz (+/-)',
    atan='atan math: std|fast|lut',
)
async def wx_set(
    interaction: discord.Interaction,
    gain: int | None = None,
    squelch: int | None = None,
    bandwidth: int | None = None,
    oversampling: int | None = None,
    ppm: int | None = None,
    deemp: bool | None = None,
    offset: bool | None = None,
    center_offset_khz: float | None = None,
    atan: str | None = None,
):
    try:
        if gain is not None: SETTINGS.gain_db = gain
        if squelch is not None: SETTINGS.squelch = max(0, squelch)
        if bandwidth is not None: SETTINGS.samplerate = max(8000, bandwidth)
        if oversampling is not None: SETTINGS.oversampling = max(1, oversampling)
        if ppm is not None: SETTINGS.ppm = ppm
        if deemp is not None: SETTINGS.deemp = bool(deemp)
        if offset is not None: SETTINGS.offset_tuning = bool(offset)
        if center_offset_khz is not None: SETTINGS.center_offset_hz = int(center_offset_khz * 1000.0)
        if atan is not None:
            if atan.lower() in ('std','fast','lut'):
                SETTINGS.atan_mode = atan.lower()
            else:
                await interaction.response.send_message('atan must be std|fast|lut', ephemeral=True)
                return
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f"Settings applied:\n```{asdict(SETTINGS)}```", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_gain', description='Quick: set tuner gain dB (rtl_fm -g)')
@app_commands.describe(db='Gain in dB')
async def wx_gain(interaction: discord.Interaction, db: int):
    try:
        SETTINGS.gain_db = db
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f'Gain set to {db} dB', ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_defaults', description='Apply baseline settings and restart if playing.')
async def wx_defaults(interaction: discord.Interaction):
    try:
        # Baseline from user-provided settings
        SETTINGS.samplerate = 12000
        SETTINGS.gain_db = 35
        SETTINGS.squelch = 1
        SETTINGS.deemp = True
        SETTINGS.offset_tuning = False
        SETTINGS.dc_block = False
        SETTINGS.fir_size = 0
        SETTINGS.ppm = 0
        SETTINGS.center_offset_hz = 0
        SETTINGS.atan_mode = 'fast'
        SETTINGS.output_stereo = True
        SETTINGS.output_clean = True
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f"Defaults applied:\n```{asdict(SETTINGS)}```", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_stereo', description='Toggle stereo upmix (output stream stays stereo).')
@app_commands.describe(on='True for stereo upmix; false for mono (output remains stereo)')
async def wx_stereo(interaction: discord.Interaction, on: bool):
    try:
        SETTINGS.output_stereo = bool(on)
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f"Stereo upmix: {'ON' if SETTINGS.output_stereo else 'OFF'}", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_clean', description='Toggle DSP cleanup (HP/LP/denoise) on/off.')
@app_commands.describe(on='True to enable cleanup; false to disable')
async def wx_clean(interaction: discord.Interaction, on: bool):
    try:
        SETTINGS.output_clean = bool(on)
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f"DSP cleanup: {'ON' if SETTINGS.output_clean else 'OFF'}", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_dc', description='Toggle DC blocking filter (rtl_fm -E dc).')
@app_commands.describe(on='True to enable DC block; false to disable')
async def wx_dc(interaction: discord.Interaction, on: bool):
    try:
        SETTINGS.dc_block = bool(on)
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f"DC block: {'ON' if SETTINGS.dc_block else 'OFF'}", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_fir', description='Set FIR downsample filter size (rtl_fm -F), allowed: 0 or 9.')
@app_commands.describe(size='0 (fast) or 9 (cleaner roll-off)')
async def wx_fir(interaction: discord.Interaction, size: int):
    try:
        if size not in (0, 9):
            await interaction.response.send_message('FIR size must be 0 or 9', ephemeral=True)
            return
        SETTINGS.fir_size = size
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await interaction.response.send_message(f'FIR size set to {size}', ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='wx_stop', description='Stop streaming and disconnect')
async def wx_stop(interaction: discord.Interaction):
    global voice_client, current_source, current_freq_hz
    try:
        if voice_client and voice_client.is_connected():
            voice_client.stop()
        if current_source:
            current_source.cleanup()
            current_source = None
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
        current_freq_hz = None
        await interaction.response.send_message('Stopped & disconnected.', ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f'Error: {e}', ephemeral=True)

@bot.tree.command(name='health', description='Health report: tools, voice stack, intents, perms, status & settings.')
async def health(interaction: discord.Interaction):
    ff = _check_ffmpeg()
    rf = _check_rtl_fm()
    py = _check_pynacl_opus()
    it = _check_intents()
    ps = await _check_perms_and_status()
    def b(x): return '✅' if x else '❌'
    lines = []
    lines += ['**Tools**',
              f"- FFmpeg: {b(ff['present'])} (path={ff['path']})\n```{ff['info']}```",
              f"- rtl_fm: {b(rf.get('present', False))}\n```{rf.get('info','')}```"]
    lines += ['\n**Python Voice Stack**',
              f"- discord.py: {py['discord_py_version']}",
              f"- PyNaCl: {b(py['pynacl'])}",
              f"- Opus loaded: {b(py['opus_loaded'])}" + (f" (lib {py.get('opus_lib')})" if py.get('opus_lib') else '')]
    if not py['opus_loaded'] and py.get('opus_tried'):
        lines.append('```tried: ' + ', '.join(py['opus_tried']) + '```')
    lines += ['\n**Gateway Intents**',
              f"- message_content: {b(it['message_content'])}",
              f"- voice_states: {b(it['voice_states'])}"]
    lines += ['\n**Voice Status & Settings**',
              f"- connected: {ps['connected']}  playing: {ps['playing']}  latency: {ps['latency']}",
              f"- channel: {ps.get('channel_active')}  freq_mhz: {ps.get('freq_mhz')}  backend: {ps.get('backend')}  device_index: {ps.get('device_index')}  channels_loaded: {ps.get('channels_loaded')}",
              f"```settings={ps.get('settings')}```"]
    await interaction.response.send_message('\n'.join(lines), ephemeral=True)

@bot.tree.command(name='help', description='Show slash command help.')
async def help_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(build_slash_help(), ephemeral=True)

# -------------------- Prefix commands --------------------
@bot.command(name='wx_join', aliases=['join', 'start'], help='Join a voice channel; "start" auto-tunes first channel if available.')
async def wx_join(ctx: commands.Context):
    global current_channel_name

    try:
        vc_id = CFG.get('voiceChannelId')
        await connect_voice(vc_id, ctx)
        # Auto-tune the first channel if available and invoked as start
        if ctx.invoked_with == 'start' and CHANNELS:
            first_name = sorted(CHANNELS.keys())[0]
            await play_noaa(CHANNELS[first_name])
            await ctx.reply(f'Joined & tuned to {first_name} ({CHANNELS[first_name]/1e6:.6f} MHz).')
        else:
            await ctx.reply('Joined voice channel.')
    except Exception as e:
        await ctx.reply(f'Join error: {e}')

@bot.command(name='chan_tune', help='Tune by channel name from channels.json or by MHz: !chan_tune "Nav Fire" | !chan_tune 154.265')
async def chan_tune_cmd(ctx: commands.Context, *, arg: str):
    global current_channel_name

    try:
        if arg in CHANNELS:
            await play_noaa(CHANNELS[arg])
    global current_channel_meta
    current_channel_meta = CHANNEL_META.get(arg, {})
            await ctx.reply(f'Tuned to {arg} ({CHANNELS[arg]/1e6:.6f} MHz)')
            return
        try:
            mhz = float(arg)
            await play_noaa(int(mhz * 1e6))
            await ctx.reply(f'Tuned to {mhz:.6f} MHz')
            return
        except Exception:
            await ctx.reply(f"Unknown channel or invalid MHz: {arg}. Try !chan_list.")
    except Exception as e:
        await ctx.reply(f'chan_tune error: {e}')

@bot.command(name='chan_list', aliases=['channels'], help='List all channels from channels.json')
async def chan_list_cmd(ctx: commands.Context):
    try:
        if not CHANNELS:
            await ctx.reply('No channels loaded. Place channels.json and run !chan_reload.')
            return
        lines = [f"- {k}: {v/1e6:.6f} MHz" for k, v in sorted(CHANNELS.items())]
        await ctx.reply('\n'.join(lines))
    except Exception as e:
        await ctx.reply(f'chan_list error: {e}')

@bot.command(name='chan_reload', help='Reload channels.json into memory')
async def chan_reload_cmd(ctx: commands.Context):
    global CHANNELS
    try:
        CHANNELS = load_channels(CHANNELS_JSON)
        await ctx.reply(f'Reloaded {len(CHANNELS)} channels from {CHANNELS_JSON}.')
    except Exception as e:
        await ctx.reply(f'chan_reload error: {e}')

# Mirror parameter commands
@bot.command(name='wx_tune', help='Tune MHz directly: !wx_tune 154.265')
async def wx_tune_cmd(ctx: commands.Context, arg: str):
    global current_channel_name

    try:
        try:
            freq_hz = int(float(arg) * 1e6)
        except ValueError:
            await ctx.reply(f'Invalid MHz: {arg}')
            return
        await play_noaa(freq_hz)
    global current_channel_meta
    current_channel_meta = None
        await ctx.reply(f'Tuned to {freq_hz/1e6:.6f} MHz')
    except Exception as e:
        await ctx.reply(f'Tune error: {e}')

@bot.command(name='wx_gain', help='Set tuner gain dB (rtl_fm -g)')
async def wx_gain_cmd(ctx: commands.Context, db: int):
    try:
        SETTINGS.gain_db = db
        await _restart_pipeline()
        await ctx.reply(f'Gain set to {db} dB')
    except Exception as e:
        await ctx.reply(f'Gain error: {e}')

@bot.command(name='wx_squelch', help='Set squelch level; 0 disables (rtl_fm -l)')
async def wx_squelch_cmd(ctx: commands.Context, level: int):
    try:
        SETTINGS.squelch = max(0, level)
        await _restart_pipeline()
        await ctx.reply(f'Squelch set to {SETTINGS.squelch}')
    except Exception as e:
        await ctx.reply(f'Squelch error: {e}')

@bot.command(name='wx_bw', help='Set demod sample rate in Hz (rtl_fm -s)')
async def wx_bw_cmd(ctx: commands.Context, hz: int):
    try:
        SETTINGS.samplerate = max(8000, hz)
        await _restart_pipeline()
        await ctx.reply(f'Demod sample rate set to {SETTINGS.samplerate} Hz')
    except Exception as e:
        await ctx.reply(f'Bandwidth error: {e}')

@bot.command(name='wx_oversampling', help='Set oversampling (rtl_fm -o), e.g., 4')
async def wx_oversampling_cmd(ctx: commands.Context, n: int):
    try:
        SETTINGS.oversampling = max(1, n)
        await _restart_pipeline()
        await ctx.reply(f'Oversampling set to {SETTINGS.oversampling}')
    except Exception as e:
        await ctx.reply(f'Oversampling error: {e}')

@bot.command(name='wx_ppm', help='Set PPM correction (rtl_fm -p)')
async def wx_ppm_cmd(ctx: commands.Context, ppm: int):
    try:
        SETTINGS.ppm = ppm
        await _restart_pipeline()
        await ctx.reply(f'PPM set to {SETTINGS.ppm}')
    except Exception as e:
        await ctx.reply(f'PPM error: {e}')

@bot.command(name='wx_deemp', help='Toggle de-emphasis (rtl_fm -E deemp): !wx_deemp on|off')
async def wx_deemp_cmd(ctx: commands.Context, state: str):
    try:
        SETTINGS.deemp = state.lower() == 'on'
        await _restart_pipeline()
        await ctx.reply(f"de-emphasis: {'ON' if SETTINGS.deemp else 'OFF'}")
    except Exception as e:
        await ctx.reply(f'deemp error: {e}')

@bot.command(name='wx_offset', help='Toggle offset tuning (rtl_fm -E offset): !wx_offset on|off')
async def wx_offset_cmd(ctx: commands.Context, state: str):
    try:
        SETTINGS.offset_tuning = state.lower() == 'on'
        await _restart_pipeline()
        await ctx.reply(f"offset tuning: {'ON' if SETTINGS.offset_tuning else 'OFF'}")
    except Exception as e:
        await ctx.reply(f'offset error: {e}')

@bot.command(name='wx_dc', help='Toggle DC blocking filter (rtl_fm -E dc): !wx_dc on|off')
async def wx_dc_cmd(ctx: commands.Context, state: str):
    try:
        SETTINGS.dc_block = state.lower() == 'on'
        await _restart_pipeline()
        await ctx.reply(f"DC block: {'ON' if SETTINGS.dc_block else 'OFF'}")
    except Exception as e:
        await ctx.reply(f'DC block error: {e}')

@bot.command(name='wx_fir', help='Set FIR downsample filter size (rtl_fm -F), allowed: 0 or 9.')
async def wx_fir_cmd(ctx: commands.Context, size: int):
    try:
        if size not in (0, 9):
            await ctx.reply('FIR size must be 0 or 9')
            return
        SETTINGS.fir_size = size
        await _restart_pipeline()
        await ctx.reply(f'FIR size set to {size}')
    except Exception as e:
        await ctx.reply(f'FIR error: {e}')

@bot.command(name='wx_centeroffset', help='Set center frequency offset in kHz (+/-)')
async def wx_centeroffset_cmd(ctx: commands.Context, khz: float):
    try:
        SETTINGS.center_offset_hz = int(khz * 1000.0)
        await _restart_pipeline()
        await ctx.reply(f'Center offset set to {khz:+.3f} kHz')
    except Exception as e:
        await ctx.reply(f'center offset error: {e}')

@bot.command(name='wx_atan', help='Set atan math (rtl_fm -A): std|fast|lut')
async def wx_atan_cmd(ctx: commands.Context, mode: str):
    try:
        if mode.lower() not in ('std', 'fast', 'lut'):
            await ctx.reply('atan must be std|fast|lut')
            return
        SETTINGS.atan_mode = mode.lower()
        await _restart_pipeline()
        await ctx.reply(f'atan mode set to {SETTINGS.atan_mode}')
    except Exception as e:
        await ctx.reply(f'atan error: {e}')

@bot.command(name='wx_stop', aliases=['leave', 'stop'], help='Stop stream and disconnect.')
async def wx_stop_cmd(ctx: commands.Context):
    global voice_client, current_source, current_freq_hz
    try:
        if voice_client and voice_client.is_connected():
            voice_client.stop()
        if current_source:
            current_source.cleanup()
            current_source = None
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
        current_freq_hz = None
        await ctx.reply('Stopped & disconnected.')
    except Exception as e:
        await ctx.reply(f'Stop error: {e}')

@bot.command(name='wx_defaults', help='Apply baseline settings and restart if playing.')
async def wx_defaults_cmd(ctx: commands.Context):
    try:
        SETTINGS.samplerate = 12000
        SETTINGS.gain_db = 35
        SETTINGS.squelch = 1
        SETTINGS.deemp = True
        SETTINGS.offset_tuning = False
        SETTINGS.dc_block = False
        SETTINGS.fir_size = 0
        SETTINGS.ppm = 0
        SETTINGS.center_offset_hz = 0
        SETTINGS.atan_mode = 'fast'
        SETTINGS.output_stereo = True
        SETTINGS.output_clean = True
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await ctx.reply(f"Defaults applied:\n```{asdict(SETTINGS)}```")
    except Exception as e:
        await ctx.reply(f'Defaults error: {e}')

@bot.command(name='wx_stereo', help='Toggle stereo upmix (on duplicates mono to L/R; off = mono upmix off). Output stays stereo.')
async def wx_stereo_cmd(ctx: commands.Context, state: str):
    try:
        SETTINGS.output_stereo = (state.lower() == 'on')
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await ctx.reply(f"Stereo upmix: {'ON' if SETTINGS.output_stereo else 'OFF'}")
    except Exception as e:
        await ctx.reply(f'Stereo toggle error: {e}')

@bot.command(name='wx_clean', help='Toggle DSP cleanup (HP/LP/denoise): !wx_clean on|off')
async def wx_clean_cmd(ctx: commands.Context, state: str):
    try:
        SETTINGS.output_clean = (state.lower() == 'on')
        if voice_client and voice_client.is_connected() and current_freq_hz:
            await _restart_pipeline()
        await ctx.reply(f"DSP cleanup: {'ON' if SETTINGS.output_clean else 'OFF'}")
    except Exception as e:
        await ctx.reply(f'wx_clean error: {e}')

@bot.command(name='health', help='Health report: tools, voice stack, intents, perms, status & settings.')
async def health_cmd(ctx: commands.Context):
    ff = _check_ffmpeg()
    rf = _check_rtl_fm()
    py = _check_pynacl_opus()
    it = _check_intents()
    ps = await _check_perms_and_status()
    def b(x): return '✅' if x else '❌'
    lines = []
    lines += ['**Tools**',
              f"- FFmpeg: {b(ff['present'])} (path={ff['path']})\n```{ff['info']}```",
              f"- rtl_fm: {b(rf.get('present', False))}\n```{rf.get('info','')}```"]
    lines += ['\n**Python Voice Stack**',
              f"- discord.py: {py['discord_py_version']}",
              f"- PyNaCl: {b(py['pynacl'])}",
              f"- Opus loaded: {b(py['opus_loaded'])}" + (f" (lib {py.get('opus_lib')})" if py.get('opus_lib') else '')]
    if not py['opus_loaded'] and py.get('opus_tried'):
        lines.append('```tried: ' + ', '.join(py['opus_tried']) + '```')
    lines += ['\n**Gateway Intents**',
              f"- message_content: {b(it['message_content'])}",
              f"- voice_states: {b(it['voice_states'])}"]
    lines += ['\n**Voice Status & Settings**',
              f"- connected: {ps['connected']}  playing: {ps['playing']}  latency: {ps['latency']}",
              f"- channel: {ps.get('channel_active')}  freq_mhz: {ps.get('freq_mhz')}  backend: {ps.get('backend')}  device_index: {ps.get('device_index')}  channels_loaded: {ps.get('channels_loaded')}",
              f"```settings={ps.get('settings')}```"]
    await ctx.reply('\n'.join(lines))

@bot.command(name='help', help='Show prefix command help.')
async def help_prefix(ctx: commands.Context):
    await ctx.reply(build_prefix_help(CFG.get('prefix', '!')))

# -------------------- Entrypoint --------------------
def main():
    token = CFG.get('token')
    if not token:
        print('[ERROR] No token found. Set DISCORD_CFG or DISCORD_TOKEN.')
        sys.exit(1)
    def _sigterm(*_):
        asyncio.get_event_loop().create_task(_shutdown())
    async def _shutdown():
        try:
            await bot.close()
        except Exception:
            pass
    signal.signal(signal.SIGTERM, _sigterm)
    bot.run(token)

if __name__ == '__main__':
    main()


# ---- CTCSS per-channel metadata ----
CHANNEL_META: dict[str, dict] = {}

def load_channels(path: str) -> dict:
    """Override: load channels and fill CHANNEL_META from JSON that may contain per-channel objects.
    Accepts either:  {"Name": 154.265, ...}  OR  {"Name":{"mhz":154.265,"ctcss_hz":192.8,"ctcss_notch":true}}
    """
    global CHANNEL_META
    CHANNEL_META = {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        out: dict[str, int] = {}
        for name, val in data.items():
            if isinstance(val, dict):
                mhz = float(val.get('mhz'))
                out[name] = int(round(mhz * 1_000_000))
                meta = {}
                if val.get('ctcss_hz') is not None:
                    meta['ctcss_hz'] = float(val['ctcss_hz'])
                meta['ctcss_notch'] = bool(val.get('ctcss_notch', False))
                CHANNEL_META[name] = meta
            else:
                out[name] = int(round(float(val) * 1_000_000))
        return out
    except Exception:
        return {}

# Refresh CHANNELS so CHANNEL_META is populated at import time
CHANNELS = load_channels(CHANNELS_JSON)



# ---- CTCSS detection (Goertzel) & Audio gating ----
import math
from collections import deque

class CTCSSDetector:
    """Goertzel-based single-tone detector for CTCSS (67–254.1 Hz)."""
    def __init__(self, target_hz: float, sample_rate: int = 48000,
                 window_ms: int = 300, power_ratio_db: float = 12.0):
        self.target_hz = float(target_hz)
        self.fs = int(sample_rate)
        self.N = int(self.fs * (window_ms / 1000.0))
        self.buf = deque(maxlen=self.N)
        k = int(0.5 + (self.N * self.target_hz / self.fs))
        self._omega = (2.0 * math.pi * k) / self.N
        self._coeff = 2.0 * math.cos(self._omega)
        self._ratio_db = float(power_ratio_db)

    def add_frame(self, stereo_s16le: bytes):
        import struct
        for L, R in struct.iter_unpack('<hh', stereo_s16le):
            self.buf.append(0.5 * (L + R))

    def _goertzel_power(self):
        s_prev = 0.0; s_prev2 = 0.0
        for x in self.buf:
            s = x + self._coeff * s_prev - s_prev2
            s_prev2 = s_prev; s_prev = s
        return (s_prev2**2 + s_prev**2 - self._coeff * s_prev * s_prev2)

    def present(self) -> bool:
        if len(self.buf) < self.N // 2:
            return False
        tone = self._goertzel_power()
        wb = sum(x*x for x in self.buf) + 1e-9
        ratio_db = 10.0 * math.log10(tone / (wb / len(self.buf)))
        return ratio_db >= self._ratio_db

# Current channel meta for gating
current_channel_meta: dict | None = None

# Override NOAAStreamSource with tone gating support
class NOAAStreamSource(discord.AudioSource):
    def __init__(self, freq_hz: int, settings: SDRSettings):
        self.base_freq_hz = freq_hz
        self.settings = settings
        self.rf_proc = None
        self.ff_proc = None
        # Prepare detector if channel has ctcss_hz
        self.ctcss = None
        self.ctcss_notch = False
        if isinstance(current_channel_meta, dict):
            ct = current_channel_meta.get('ctcss_hz')
            if ct is not None:
                self.ctcss = CTCSSDetector(ct)
            self.ctcss_notch = bool(current_channel_meta.get('ctcss_notch', False))
        self._start_pipeline()

    @property
    def freq_hz(self) -> int:
        return self.base_freq_hz + int(self.settings.center_offset_hz or 0)

    def _start_pipeline(self):
        rf_cmd = [
            RTL_FM_EXE,
            '-f', str(self.freq_hz),
            '-M', 'fm',
            '-s', str(self.settings.samplerate),
            '-g', str(self.settings.gain_db),
            '-l', str(self.settings.squelch),
            '-o', str(self.settings.oversampling),
            '-A', self.settings.atan_mode,
        ]
        if self.settings.deemp:
            rf_cmd += ['-E', 'deemp']
        if self.settings.offset_tuning:
            rf_cmd += ['-E', 'offset']
        if self.settings.dc_block:
            rf_cmd += ['-E', 'dc']
        if self.settings.fir_size in (0, 9):
            rf_cmd += ['-F', str(self.settings.fir_size)]
        if self.settings.ppm:
            rf_cmd += ['-p', str(self.settings.ppm)]
        if RTL_DEVICE_INDEX not in (None, ''):
            rf_cmd += ['-d', str(RTL_DEVICE_INDEX)]
        rf_cmd += ['-']

        stereo_pan = 'pan=stereo;c0=c0;c1=c0'
        # Lower high-pass cutoff if tone gating enabled so CTCSS survives
        hp_cut = 40 if self.ctcss else 300
        clean_chain = f'highpass=f={hp_cut},lowpass=f=4000,afftdn=nr=15:nf=-30:tn=1'
        af_chain = f'{stereo_pan},{clean_chain}'
        if self.ctcss and self.ctcss_notch and getattr(self.ctcss, 'target_hz', None):
            af_chain = f"{af_chain},anequalizer=f={self.ctcss.target_hz}:t=o:w=1"

        ff_cmd = [
            FFMPEG_EXE, '-hide_banner', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(self.settings.samplerate), '-ac', '1', '-i', 'pipe:0',
            '-af', af_chain,
            '-ac', '2', '-ar', '48000', '-f', 's16le', 'pipe:1'
        ]

        self.rf_proc = subprocess.Popen(
            rf_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW
        )
        self.ff_proc = subprocess.Popen(
            ff_cmd, stdin=self.rf_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW
        )

    def _frame_len_bytes(self) -> int:
        return SAMPLES_PER_20MS * 2 * 2

    def read(self):
        try:
            need = self._frame_len_bytes()
            data = self.ff_proc.stdout.read(need)
            if not data or len(data) < need:
                return b''
            if self.ctcss:
                self.ctcss.add_frame(data)
                if not self.ctcss.present():
                    return b'\x00' * need
            return data
        except Exception:
            return b''

    def cleanup(self):
        for p in (self.ff_proc, self.rf_proc):
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass

    def is_opus(self):
        return False

