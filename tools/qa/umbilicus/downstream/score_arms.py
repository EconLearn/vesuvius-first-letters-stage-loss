"""Score downstream spiral fits of the umbilicus arms. For each arm: winding consistency on the organiser surfaces
(the JSON written by tools/surf_wind.py; share_mode = fraction of text pixels on the modal fitted winding, dv = winding
gap w060 - w058, truth 2.0) and the fit's own track satisfaction (last 'satisfied_tracks' / 'satisfied_track_points'
lines of its log). usage: score_arms.py OUT.json ARM=WIND_JSON,FIT_LOG [ARM=...]   (WIND_JSON may be '-')"""
import json, re, sys
from pathlib import Path
rows = {}
for spec in sys.argv[2:]:
    arm, files = spec.split('=', 1); wind, flog = files.split(',', 1); r = {}
    t = Path(flog).read_text(errors='ignore') if Path(flog).exists() else ''
    for key in ('satisfied_tracks', 'satisfied_track_points'):
        m = re.findall(key + r' = \d+/\d+ \(([\d.]+)%\)', t)
        if m: r[key] = float(m[-1])
    if wind != '-' and Path(wind).exists():
        W = json.loads(Path(wind).read_text()).get('windings', {}); sm, p50 = [], {}
        for k in ('w058', 'w060', 'w062'):
            tx = (W.get(k) or {}).get('text') or {}
            if tx.get('share_mode') is not None: sm.append(round(tx['share_mode'], 4))
            if tx.get('v_p5_p50_p95'): p50[k] = tx['v_p5_p50_p95'][1]
        r.update(share_modes=sm, share_mode_mean=round(sum(sm) / len(sm), 4) if sm else None)
        if 'w058' in p50 and 'w060' in p50: r['dv'] = round(p50['w060'] - p50['w058'], 2)
    rows[arm] = r; print(arm, r)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
