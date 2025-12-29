
# -*- coding: utf-8 -*-
import os, sys, asyncio, signal, subprocess, shutil, json, math, logging
from dataclasses import dataclass, asdict
from collections import deque
import discord
from discord import app_commands
from discord.ext import commands

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('radio-bot')

# ------------------ Config ------------------
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
        if gid: out['guildId'] = int(gid)
    except Exception: pass
    try:
        if vid: out['voiceChannelId'] = int(vid)
    except Exception: pass
    return out

CFG = parse_cfg()

# ------------------ Executables ------------------
DEFAULT_FFMPEG = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
DEFAULT_RTLFM  = 'rtl_fm.exe' if os.name == 'nt' else 'rtl_fm'
FFMPEG_EXE     = os.getenv('FFMPEG_EXE', DEFAULT_FFMPEG)
RTL_FM_EXE     = os.getenv('RTL_FM_EXE', DEFAULT_RTLFM)
CHANNELS_JSON  = os.getenv('CHANNELS_JSON', 'channels.json')
RTL_DEVICE_INDEX = os.getenv('RTL_DEVICE_INDEX')
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

# ------------------ Optional DTCS map (for verification) ------------------
DTCS_MAP_PATH = os.getenv('DTCS_MAP', 'dcs_map.json')
DTCS_MAP = {}
if os.path.isfile(DTCS_MAP_PATH):
    try:
        with open(DTCS_MAP_PATH, 'r') as f:
            DTCS_MAP = json.load(f)
    except Exception:
        DTCS_MAP = {}

# ------------------ Channel meta ------------------
CHANNEL_META = {}

def load_channels(path: str) -> dict:
    global CHANNEL_META
    CHANNEL_META = {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        out = {}
        for name, val in data.items():
            if isinstance(val, dict):
                mhz = float(val.get('mhz'))
                out[name] = int(round(mhz * 1_000_000))
                meta = {}
                squ = val.get('squelch') or {}
                if squ.get('type'):
                    meta['type'] = str(squ['type'])
                # Base CTCSS
                if squ.get('ctcss_hz') is not None:
                    meta['ctcss_hz'] = float(squ['ctcss_hz'])
                if squ.get('ctcss_notch') is not None:
                    meta['ctcss_notch'] = bool(squ['ctcss_notch'])
                # DTCS
                if squ.get('dtcs_code') is not None:
                    meta['dtcs_code'] = int(squ['dtcs_code'])
                if squ.get('dtcs_polarity') is not None:
                    meta['dtcs_polarity'] = str(squ['dtcs_polarity'])
                # Cross
                if squ.get('rx'):
                    meta['rx'] = squ['rx']
                    if isinstance(squ['rx'], dict):
                        if squ['rx'].get('ctcss_hz') is not None:
                            meta['ctcss_hz'] = float(squ['rx']['ctcss_hz'])
                            if meta.get('ctcss_notch') is None:
                                meta['ctcss_notch'] = True
                        if squ['rx'].get('dtcs_code') is not None:
                            meta['rx_dtcs_code'] = int(squ['rx']['dtcs_code'])
                            meta['rx_dtcs_polarity'] = str(squ['rx'].get('dtcs_polarity',''))
                if squ.get('tx'):
                    meta['tx'] = squ['tx']
                # Verification toggle
                if squ.get('dtcs_verify') is not None:
                    meta['dtcs_verify'] = bool(squ['dtcs_verify'])
                CHANNEL_META[name] = meta
            else:
                out[name] = int(round(float(val) * 1_000_000))
        return out
    except Exception as e:
        log.error('load_channels error: {}'.format(e))
        return {}

CHANNELS = load_channels(CHANNELS_JSON)

# ------------------ SDR Settings ------------------
@dataclass
class SDRSettings:
    gain_db: int = 35
    squelch: int = 1
    samplerate: int = 12000
    oversampling: int = 4
    ppm: int = 0
    atan_mode: str = 'fast'
    deemp: bool = True
    offset_tuning: bool = False
    dc_block: bool = False
    fir_size: int = 0
    center_offset_hz: int = 0
    output_stereo: bool = True
    output_clean: bool = True

SETTINGS = SDRSettings()
SAMPLES_PER_20MS = 960

# ------------------ Detectors ------------------
class CTCSSDetector:
    def __init__(self, target_hz: float, sample_rate: int = 48000, window_ms: int = 300, power_ratio_db: float = 12.0):
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

class DCSDetector:
    """DTCS presence + optional code/polarity verification (requires dcs_map.json).
    Presence: energy around ~134.4Hz and ~268.8Hz using Goertzel components.
    Verify: normalized cross-correlation against a learned signature (numpy optional).
    """
    def __init__(self, sample_rate: int = 48000, window_ms: int = 300, power_ratio_db: float = 10.0, verify=False, code=None, polarity=''):
        self.fs = int(sample_rate)
        self.N = int(self.fs * (window_ms / 1000.0))
        self.buf = deque(maxlen=self.N)
        self.verify = bool(verify)
        self.code = None if code is None else int(code)
        self.polarity = str(polarity or '')
        def coeff_for(f):
            k = int(0.5 + (self.N * f / self.fs))
            omega = (2.0 * math.pi * k) / self.N
            return 2.0 * math.cos(omega)
        self.c1 = coeff_for(134.4)
        self.c2 = coeff_for(268.8)
        self._ratio_db = float(power_ratio_db)
        self.sig = None
        if self.verify and self.code is not None:
            key = f"{self.code}:{self.polarity or 'NN'}"
            sig = DTCS_MAP.get(key)
            if isinstance(sig, list) and len(sig) > 0:
                self.sig = [float(x) for x in sig]
    def add_frame(self, stereo_s16le: bytes):
        import struct
        for L, R in struct.iter_unpack('<hh', stereo_s16le):
            self.buf.append(0.5 * (L + R))
    def _power(self, coeff):
        s_prev = 0.0; s_prev2 = 0.0
        for x in self.buf:
            s = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev; s_prev = s
        return (s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2)
    def _present(self) -> bool:
        if len(self.buf) < self.N // 2:
            return False
        p1 = self._power(self.c1)
        p2 = self._power(self.c2)
        wb = sum(x*x for x in self.buf) + 1e-9
        ratio_db = 10.0 * math.log10(((p1 + p2) / 2.0) / (wb / len(self.buf)))
        return ratio_db >= self._ratio_db
    def _verify(self) -> bool:
        if not self.sig or len(self.buf) < len(self.sig):
            return False
        # Try numpy; if unavailable, fallback to pure Python
        try:
            import numpy as np
            x = np.array(self.sig, dtype=float)
            y = np.array(list(self.buf)[-len(self.sig):], dtype=float)
            x = (x - x.mean()) / (x.std() + 1e-9)
            y = (y - y.mean()) / (y.std() + 1e-9)
            corr = float(np.dot(x, y) / len(x))
        except Exception:
            # Pure Python normalization and dot
            sig = self.sig[-len(self.sig):]
            buf = list(self.buf)[-len(sig):]
            xm = sum(sig)/len(sig)
            ym = sum(buf)/len(buf)
            xs = math.sqrt(sum((s - xm)**2 for s in sig)/len(sig)) + 1e-9
            ys = math.sqrt(sum((b - ym)**2 for b in buf)/len(buf)) + 1e-9
            xnorm = [(s - xm)/xs for s in sig]
            ynorm = [(b - ym)/ys for b in buf]
            corr = sum(x*y for x,y in zip(xnorm, ynorm))/len(xnorm)
        thr = float(os.getenv('DTCS_VERIFY_THRESH', '0.35'))
        return corr >= thr
    def present(self) -> bool:
        if not self._present():
            return False
        if self.verify and self.sig is not None:
            return self._verify()
        return True

# ------------------ Audio pipeline ------------------
voice_client = None
current_source = None
current_freq_hz = None
current_channel_meta = None

class NOAAStreamSource(discord.AudioSource):
    def __init__(self, freq_hz: int, settings: SDRSettings):
        self.base_freq_hz = freq_hz
        self.settings = settings
        self.rf_proc = None
        self.ff_proc = None
        self.ctcss = None
        self.ctcss_notch = False
        self.dcs = None
        mode = None
        if isinstance(current_channel_meta, dict):
            mode = str(current_channel_meta.get('type') or '').lower()
            ct = current_channel_meta.get('ctcss_hz')
            if ct is not None and mode in ('tone','tsql','cross'):
                self.ctcss = CTCSSDetector(ct)
                self.ctcss_notch = bool(current_channel_meta.get('ctcss_notch', False))
            if mode == 'dtcs' or (mode == 'cross' and current_channel_meta.get('rx_dtcs_code') is not None):
                verify = bool(current_channel_meta.get('dtcs_verify', False))
                code = current_channel_meta.get('dtcs_code') or current_channel_meta.get('rx_dtcs_code')
                pol  = current_channel_meta.get('dtcs_polarity') or current_channel_meta.get('rx_dtcs_polarity')
                self.dcs = DCSDetector(verify=verify, code=code, polarity=pol)
        self._start_pipeline()
    @property
    def freq_hz(self) -> int:
        return self.base_freq_hz + int(self.settings.center_offset_hz or 0)
    def _start_pipeline(self):
        rf_cmd = [
            RTL_FM_EXE, '-f', str(self.freq_hz), '-M', 'fm',
            '-s', str(self.settings.samplerate),
            '-g', str(self.settings.gain_db),
            '-l', str(self.settings.squelch),
            '-o', str(self.settings.oversampling),
            '-A', self.settings.atan_mode,
        ]
        if self.settings.deemp: rf_cmd += ['-E', 'deemp']
        if self.settings.offset_tuning: rf_cmd += ['-E', 'offset']
        if self.settings.dc_block: rf_cmd += ['-E', 'dc']
        if self.settings.fir_size in (0, 9): rf_cmd += ['-F', str(self.settings.fir_size)]
        if self.settings.ppm: rf_cmd += ['-p', str(self.settings.ppm)]
        if RTL_DEVICE_INDEX not in (None, ''): rf_cmd += ['-d', str(RTL_DEVICE_INDEX)]
        rf_cmd += ['-']
        # Lower HP when tone/dcs gating is active
        stereo_pan = 'pan=stereo;c0=c0;c1=c0'
        hp_cut = 35 if (self.ctcss or self.dcs) else 300
        clean_chain = f'highpass=f={hp_cut},lowpass=f=4000,afftdn=nr=15:nf=-30:tn=1'
        af_chain = f'{stereo_pan},{clean_chain}'
        if self.ctcss and self.ctcss_notch:
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
                    return b' ' * need
            elif self.dcs:
                self.dcs.add_frame(data)
                if not self.dcs.present():
                    return b' ' * need
            return data
        except Exception:
            return b''
    def cleanup(self):
        for p in (self.ff_proc, self.rf_proc):
            if p and p.poll() is None:
                try: p.terminate()
                except Exception: pass
    def is_opus(self):
        return False

# ------------------ Bot ------------------
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix=CFG.get('prefix', '!'), intents=intents)
# remove discord.py built-in prefix help to allow our custom !help
bot.remove_command('help')

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


# Error handlers
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    log.error('[SLASH-ERR] {}'.format(error))
    try:
        if interaction.response.is_done():
            await interaction.followup.send("Error: {}".format(error), ephemeral=True)
        else:
            await interaction.response.send_message("Error: {}".format(error), ephemeral=True)
    except Exception:
        pass

@bot.event
async def on_interaction(interaction: discord.Interaction):
    # Lightweight visibility for incoming interactions
    try:
        name = None
        if hasattr(interaction, 'data') and isinstance(interaction.data, dict):
            name = interaction.data.get('name')
        log.info("[INT] user={} name={} responded={}".format(getattr(interaction.user,'id',None), name, interaction.response.is_done()))
    except Exception:
        pass

# ------------------ Voice helpers ------------------
async def connect_voice(channel_id: int | None, ctx: commands.Context | None = None):
    global voice_client
    chan_obj = None
    if channel_id:
        chan = await bot.fetch_channel(channel_id)
        if isinstance(chan, discord.VoiceChannel): chan_obj = chan
        else: raise RuntimeError('VOICE_CHANNEL_ID is not a voice channel')
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
    if not voice_client or not voice_client.is_connected(): raise RuntimeError('Not connected to a voice channel')
    if current_freq_hz is None: raise RuntimeError('No frequency set; use tune command')
    if voice_client.is_playing(): voice_client.stop()
    if current_source: current_source.cleanup()
    current_source = NOAAStreamSource(freq_hz=current_freq_hz, settings=SETTINGS)
    voice_client.play(current_source, after=lambda e: log.error('[ERROR] Player: {}'.format(e)) if e else None)

async def play_noaa(freq_hz: int):
    global current_freq_hz
    current_freq_hz = freq_hz
    await _restart_pipeline()

# ------------------ Slash commands ------------------
@bot.tree.command(name='ping', description='Health check')
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong", ephemeral=True)

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

@bot.tree.command(name='help', description='Show radio bot commands')
async def help_cmd(interaction: discord.Interaction):
    text = (
        """
**Commands**
• /help — Show this help
• /ping — Health check (instant)
• /join — Join your current voice channel (or configured VOICE_CHANNEL_ID)
• /chan_list — List all channels
• /chan_tune name:<channel> | mhz:<freq> — Tune the SDR
• /chan_reload — Reload channels.json

**Notes**
Use /join before tuning if the bot is not yet in voice.
        """
    )
    await interaction.response.send_message(text, ephemeral=True)

@bot.tree.command(name='join', description='Join your current voice channel or the configured VOICE_CHANNEL_ID')
async def join(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True)
        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        if member and member.voice and member.voice.channel:
            chan = member.voice.channel
            await connect_voice(chan.id)
            await interaction.followup.send("Joined {}.".format(chan.name))
            return
        vc_id = CFG.get('voiceChannelId')
        if vc_id:
            await connect_voice(vc_id)
            chan = await bot.fetch_channel(vc_id)
            await interaction.followup.send("Joined {}.".format(chan.name))
        else:
            await interaction.followup.send("You are not in a voice channel and VOICE_CHANNEL_ID is not set.")
    except Exception as e:
        await interaction.followup.send("Join error: {}".format(e))

@bot.tree.command(name='chan_list', description='List channels')
async def chan_list(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    if not CHANNELS:
        await interaction.followup.send("No channels loaded.")
        return
    lines = ["- {}: {:.6f} MHz".format(k, v/1_000_000) for k, v in sorted(CHANNELS.items())]
    await interaction.followup.send("\n".join(lines))

@bot.tree.command(name='chan_tune', description='Tune by channel name or MHz')
@app_commands.describe(name='channel name', mhz='frequency in MHz')
async def chan_tune(interaction: discord.Interaction, name: str = None, mhz: float = None):
    global current_channel_meta
    try:
        await interaction.response.defer(ephemeral=True)
        if name:
            if name not in CHANNELS:
                await interaction.followup.send("Unknown channel '{}'.".format(name))
                return
            current_channel_meta = CHANNEL_META.get(name, {})
            await play_noaa(CHANNELS[name])
            await interaction.followup.send("Tuned to {} ({:.6f} MHz)".format(name, CHANNELS[name]/1_000_000))
        elif mhz is not None:
            current_channel_meta = None
            await play_noaa(int(mhz*1_000_000))
            await interaction.followup.send("Tuned to {:.6f} MHz".format(mhz))
        else:
            await interaction.followup.send("Provide a channel name or MHz.")
    except Exception as e:
        await interaction.followup.send("Error: {}".format(e))

@bot.tree.command(name='chan_reload', description='Reload channels.json')
async def chan_reload(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    global CHANNELS
    CHANNELS = load_channels(CHANNELS_JSON)
    await interaction.followup.send("Reloaded {} channels.".format(len(CHANNELS)))

@bot.tree.command(name='sync', description='Sync slash commands (guild/global)')
@app_commands.describe(scope='guild or global')
async def sync(interaction: discord.Interaction, scope: str = 'guild'):
    try:
        if scope.lower() == 'guild' and CFG.get('guildId'):
            await bot.tree.sync(guild=discord.Object(id=CFG['guildId']))
            await interaction.response.send_message('Slash commands synced to guild {}.'.format(CFG['guildId']), ephemeral=True)
        else:
            await bot.tree.sync()
            await interaction.response.send_message('Global slash commands synced. (may take up to ~1 hour to propagate)', ephemeral=True)
    except Exception as e:
        await interaction.response.send_message('Sync error: {}'.format(e), ephemeral=True)

# ------------------ Prefix commands ------------------
@bot.command(name='help')
async def legacy_help(ctx: commands.Context):
    await ctx.reply("Use /help for slash commands. Prefix commands: !join, !chan_tune <name|MHz>")

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

@bot.command(name='join')
async def legacy_join(ctx: commands.Context):
    try:
        vc_id = CFG.get('voiceChannelId')
        await connect_voice(vc_id, ctx)
        await ctx.reply("Joined voice channel.")
    except Exception as e:
        await ctx.reply("Join error: {}".format(e))

@bot.command(name='chan_tune')
async def chan_tune_cmd(ctx: commands.Context, *, arg: str):
    global current_channel_meta
    try:
        if arg in CHANNELS:
            current_channel_meta = CHANNEL_META.get(arg, {})
            await play_noaa(CHANNELS[arg])
            await ctx.reply("Tuned to {} ({:.6f} MHz)".format(arg, CHANNELS[arg]/1_000_000)); return
        try:
            mhz = float(arg)
            current_channel_meta = None
            await play_noaa(int(mhz * 1_000_000))
            await ctx.reply("Tuned to {:.6f} MHz".format(mhz)); return
        except Exception:
            await ctx.reply("Unknown channel or invalid MHz: {}".format(arg))
    except Exception as e:
        await ctx.reply("chan_tune error: {}".format(e))

# ------------------ Ready ------------------
@bot.event
async def on_ready():
    log.info("Logged in as {} (id={})".format(bot.user, bot.user.id))
    gid = CFG.get('guildId')
    try:
        if gid:
            await bot.tree.sync(guild=discord.Object(id=gid))
            log.info("Slash commands synced to guild {}".format(gid))
        else:
            await bot.tree.sync()
            log.info('Global slash commands synced (may take up to ~1 hour to propagate).')
    except Exception as e:
        log.error('Sync error: {}'.format(e))

# ------------------ Entrypoint ------------------
def main():
    token = CFG.get('token')
    if not token:
        log.error('No token found. Set DISCORD_CFG or DISCORD_TOKEN.')
        sys.exit(1)
    def _sigterm(*_): asyncio.get_event_loop().create_task(_shutdown())
    async def _shutdown():
        try: await bot.close()
        except Exception: pass
    signal.signal(signal.SIGTERM, _sigterm)
    bot.run(token)

if __name__ == '__main__':
    main()