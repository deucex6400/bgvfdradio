# -*- coding: utf-8 -*-
"""Learn DTCS code/polarity signatures from mono WAV recordings.
Usage:
  python dcs_learn.py --wav path.wav --code 411 --pol NN --out dcs_map.json

Record ~10 seconds of clean audio containing only the target DTCS.
The script band-limits, extracts the low-frequency baseband, and stores a normalized snippet as the signature.
"""
import argparse, json, os, sys
import numpy as np
import wave

parser = argparse.ArgumentParser()
parser.add_argument('--wav', required=True)
parser.add_argument('--code', type=int, required=True)
parser.add_argument('--pol', default='NN')
parser.add_argument('--out', default='dcs_map.json')
args = parser.parse_args()

# Load WAV
with wave.open(args.wav, 'rb') as w:
    ch = w.getnchannels(); sr = w.getframerate(); sampwidth = w.getsampwidth(); n = w.getnframes()
    if sr != 48000:
        print(f'[WARN] Expected 48000 Hz, got {sr}. Resample externally for best results.')
    raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)

# Normalize and band-limit (very simple HP/LP via FFT masking)
N = len(data)
fft = np.fft.rfft(data)
freqs = np.fft.rfftfreq(N, d=1.0/sr)
mask = (freqs >= 20) & (freqs <= 500)
fft[~mask] = 0
flt = np.fft.irfft(fft, n=N)
flt = (flt - flt.mean()) / (flt.std() + 1e-9)

# Take middle 2 seconds as signature window
win = int(sr * 2.0)
start = max(0, (N//2) - (win//2))
sig = flt[start:start+win]

# Save to map
key = f"{args.code}:{args.pol}".upper()
try:
    with open(args.out, 'r') as f:
        m = json.load(f)
except Exception:
    m = {}
m[key] = sig.tolist()
with open(args.out, 'w') as f:
    json.dump(m, f)
print(f'[OK] Stored signature for {key} in {args.out} (len={len(sig)})')
