# Portal Integration Patch — DTCS Verify

Add/keep the endpoints from the previous patch. No changes needed for API.
The editor now toggles `dtcs_verify` in the per-channel `squelch` object.
Your bot will attempt DTCS **code/polarity verification** if `dtcs_verify=true` and `dcs_map.json` contains a signature for that code (key format: `"<code>:<polarity>"`).

To learn signatures:

```bash
python dcs_learn.py --wav NAV_MED.wav --code 411 --pol NN --out dcs_map.json
```

Repeat for each code/polarity you care about. Place `dcs_map.json` beside `sdrbot.py`.
Optionally adjust the match threshold via env var:

```bash
set DTCS_VERIFY_THRESH=0.40  # Windows
export DTCS_VERIFY_THRESH=0.40  # Linux
```
