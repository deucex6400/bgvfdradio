# -*- coding: utf-8 -*-
import os, json
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import sdrbot

API_KEY = os.getenv('PORTAL_API_KEY')
SECRET_KEY = os.getenv('PORTAL_SECRET', 'change-this-in-prod-please')
PORTAL_USER = os.getenv('PORTAL_USER', 'admin')
PORTAL_PASS = os.getenv('PORTAL_PASS', '3400')

app = FastAPI(title='SDR Bot Portal')
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount('/static', StaticFiles(directory='portal/static'), name='static')
templates = Jinja2Templates(directory='portal/templates')

def is_logged_in(request: Request) -> bool:
    return (request.session.get('user') == PORTAL_USER)

async def require_login(request: Request):
    if not is_logged_in(request):
        raise HTTPException(status_code=401, detail='login required')

async def require_api_or_login(request: Request):
    if is_logged_in(request):
        return True
    if not API_KEY:
        raise HTTPException(status_code=401, detail='login required')
    supplied = request.headers.get('X-API-Key') or request.query_params.get('api_key')
    if supplied != API_KEY:
        raise HTTPException(status_code=401, detail='unauthorized')
    return True

# Channel view helper
def mhz_view() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, hz in sdrbot.CHANNELS.items():
        entry: Dict[str, Any] = {'mhz': round(hz / 1e6, 6)}
        meta = sdrbot.CHANNEL_META.get(name, {})
        if meta.get('ctcss_hz') is not None:
            entry['ctcss_hz'] = float(meta['ctcss_hz'])
        entry['ctcss_notch'] = bool(meta.get('ctcss_notch', False))
        out[name] = entry
    return out

@app.get('/login', response_class=HTMLResponse)
async def login_get(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url='/', status_code=302)
    return templates.TemplateResponse('login.html', {'request': request})

@app.post('/login', response_class=HTMLResponse)
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == PORTAL_USER and password == PORTAL_PASS:
        request.session['user'] = PORTAL_USER
        return RedirectResponse(url='/', status_code=302)
    return templates.TemplateResponse('login.html', {'request': request, 'error': 'Invalid username or password'})

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/login', status_code=302)

@app.get('/', response_class=HTMLResponse)
async def dashboard(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url='/login', status_code=302)
    ff = sdrbot._check_ffmpeg(); rf = sdrbot._check_rtl_fm(); py = sdrbot._check_pynacl_opus(); it = sdrbot._check_intents(); ps = await sdrbot._check_perms_and_status()
    channels = mhz_view()
    return templates.TemplateResponse('index.html', {'request': request, 'ff': ff, 'rf': rf, 'py': py, 'it': it, 'ps': ps, 'channels': channels, 'cfg': sdrbot.CFG, 'has_api_key': bool(API_KEY)})

@app.get('/api/health')
async def api_health(request: Request, _auth: Any = Depends(require_api_or_login)):
    ff = sdrbot._check_ffmpeg(); rf = sdrbot._check_rtl_fm(); py = sdrbot._check_pynacl_opus(); it = sdrbot._check_intents(); ps = await sdrbot._check_perms_and_status()
    return JSONResponse({'ffmpeg': ff, 'rtl_fm': rf, 'python_voice': py, 'intents': it, 'status': ps})

@app.post('/api/join')
async def api_join(request: Request, _auth: Any = Depends(require_api_or_login)):
    try:
        await sdrbot.connect_voice(sdrbot.CFG.get('voiceChannelId'))
        return {'ok': True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/api/leave')
async def api_leave(request: Request, _auth: Any = Depends(require_api_or_login)):
    try:
        if sdrbot.voice_client and sdrbot.voice_client.is_connected(): sdrbot.voice_client.stop()
        if sdrbot.current_source: sdrbot.current_source.cleanup(); sdrbot.current_source = None
        if sdrbot.voice_client and sdrbot.voice_client.is_connected(): await sdrbot.voice_client.disconnect()
        sdrbot.current_freq_hz = None
        return {'ok': True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/api/tune')
async def api_tune(request: Request, name: Optional[str] = Form(None), mhz: Optional[float] = Form(None), _auth: Any = Depends(require_api_or_login)):
    try:
        if name:
            if name not in sdrbot.CHANNELS: raise HTTPException(status_code=404, detail=f"Unknown channel '{name}'")
            sdrbot.current_channel_meta = sdrbot.CHANNEL_META.get(name, {})
            await sdrbot.play_noaa(sdrbot.CHANNELS[name])
            return {'ok': True, 'tuned': {'name': name}}
        if mhz is not None:
            sdrbot.current_channel_meta = None
            await sdrbot.play_noaa(int(float(mhz)*1e6))
            return {'ok': True, 'tuned': {'mhz': round(float(mhz),6)}}
        raise HTTPException(status_code=400, detail='Provide name or mhz')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/api/settings')
async def api_settings(request: Request,
    gain: Optional[int] = Form(None),
    squelch: Optional[int] = Form(None),
    bandwidth: Optional[int] = Form(None),
    oversampling: Optional[int] = Form(None),
    ppm: Optional[int] = Form(None),
    deemp: Optional[bool] = Form(None),
    offset: Optional[bool] = Form(None),
    center_offset_khz: Optional[float] = Form(None),
    atan: Optional[str] = Form(None),
    stereo: Optional[bool] = Form(None),
    clean: Optional[bool] = Form(None),
    dc: Optional[bool] = Form(None),
    fir: Optional[int] = Form(None),
    _auth: Any = Depends(require_api_or_login),
):
    try:
        if gain is not None: sdrbot.SETTINGS.gain_db = gain
        if squelch is not None: sdrbot.SETTINGS.squelch = max(0, squelch)
        if bandwidth is not None: sdrbot.SETTINGS.samplerate = max(8000, bandwidth)
        if oversampling is not None: sdrbot.SETTINGS.oversampling = max(1, oversampling)
        if ppm is not None: sdrbot.SETTINGS.ppm = ppm
        if deemp is not None: sdrbot.SETTINGS.deemp = bool(deemp)
        if offset is not None: sdrbot.SETTINGS.offset_tuning = bool(offset)
        if center_offset_khz is not None: sdrbot.SETTINGS.center_offset_hz = int(center_offset_khz * 1000.0)
        if atan is not None:
            if atan.lower() in ('std','fast','lut'): sdrbot.SETTINGS.atan_mode = atan.lower()
            else: raise HTTPException(status_code=400, detail='atan must be std/fast/lut')
        if stereo is not None: sdrbot.SETTINGS.output_stereo = bool(stereo)
        if clean is not None: sdrbot.SETTINGS.output_clean = bool(clean)
        if dc is not None: sdrbot.SETTINGS.dc_block = bool(dc)
        if fir is not None:
            if fir not in (0,9): raise HTTPException(status_code=400, detail='FIR size must be 0 or 9')
            sdrbot.SETTINGS.fir_size = fir
        if sdrbot.voice_client and sdrbot.voice_client.is_connected() and sdrbot.current_freq_hz:
            await sdrbot._restart_pipeline()
        return {'ok': True, 'settings': sdrbot.asdict(sdrbot.SETTINGS)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/api/channels')
async def api_channels(request: Request, _auth: Any = Depends(require_api_or_login)):
    return {'path': sdrbot.CHANNELS_JSON, 'count': len(sdrbot.CHANNELS), 'channels': mhz_view()}

@app.post('/api/channels/reload')
async def api_channels_reload(request: Request, _auth: Any = Depends(require_api_or_login)):
    try:
        sdrbot.CHANNELS = sdrbot.load_channels(sdrbot.CHANNELS_JSON)
        return {'ok': True, 'count': len(sdrbot.CHANNELS)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/api/channels/append')
async def api_channels_append(request: Request,
    name: str = Form(...), mhz: float = Form(...), ctcss_hz: float | None = Form(None), ctcss_notch: bool | None = Form(None),
    _auth: Any = Depends(require_api_or_login),
):
    try:
        cur = mhz_view()
        entry: Dict[str, Any] = {'mhz': round(float(mhz), 6)}
        if ctcss_hz is not None: entry['ctcss_hz'] = float(ctcss_hz)
        if ctcss_notch is not None: entry['ctcss_notch'] = bool(ctcss_notch)
        cur[name] = entry
        with open(sdrbot.CHANNELS_JSON, 'w') as f:
            json.dump(cur, f, indent=2)
        sdrbot.CHANNELS = sdrbot.load_channels(sdrbot.CHANNELS_JSON)
        return {'ok': True, 'count': len(sdrbot.CHANNELS), 'added': name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put('/api/channels')
async def api_channels_replace(request: Request, payload: Dict[str, Any], _auth: Any = Depends(require_api_or_login)):
    try:
        normalized: Dict[str, Any] = {}
        for name, val in payload.items():
            if isinstance(val, dict):
                mhz = float(val['mhz'])
                normalized[name] = {'mhz': round(mhz, 6)}
                if val.get('ctcss_hz') is not None:
                    normalized[name]['ctcss_hz'] = float(val['ctcss_hz'])
                if val.get('ctcss_notch') is not None:
                    normalized[name]['ctcss_notch'] = bool(val['ctcss_notch'])
            else:
                normalized[name] = round(float(val), 6)
        with open(sdrbot.CHANNELS_JSON, 'w') as f:
            json.dump(normalized, f, indent=2)
        sdrbot.CHANNELS = sdrbot.load_channels(sdrbot.CHANNELS_JSON)
        return {'ok': True, 'count': len(sdrbot.CHANNELS)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
