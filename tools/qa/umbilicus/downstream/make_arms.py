"""Umbilicus arms for the downstream spiral-fit test: hand (Drobkov), unc (patched VC3D estimator, uncrush 1.5), orig
(uncrush 0), shift (hand + 300 voxels in x, the sabotage arm). The estimator runs on the published m7 normal grids every
100 z from Z0 to Z1, 9 seeds, median, then a running median over 5 slices.
usage: make_arms.py UMB_DIR HARNESS OUT_DIR SCROLL Z0 Z1 PREFIX
  UMB_DIR: the benchmark directory of ../fetch_meta.py (scrolls.json, ref/); HARNESS: vc3d_harness from ../vc3d/build.sh"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, s3fs
U, H, O = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); S, Z0, Z1, PRE = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
O.mkdir(parents=True, exist_ok=True); G = O / f'grids_{PRE}'; G.mkdir(exist_ok=True)
SC = json.loads((U / 'scrolls.json').read_text())[S]; ng = SC['m7'].replace('.zarr', '.normal-grids'); fs = s3fs.S3FileSystem(anon=True); vox = SC['vox_um'] / 1000
ZS = list(range(Z0, Z1 + 1, 100)); files = []
for z in ZS:
    f = G / f'{z:06d}.grid'
    if not f.exists():
        for att in range(5):
            try: f.write_bytes(fs.cat_file(f'{ng}/xy/{z:06d}.grid')); break
            except Exception: time.sleep(5 * (att + 1))
    files.append(str(f))
def est(a):
    E = [[list(map(float, l.split())) for l in subprocess.run([H, str(a), str(sd)] + files, capture_output=True, text=True).stdout.split('\n') if l.strip()] for sd in range(1, 10)]
    raw = np.median(np.array(E), 0); return np.array([np.median(raw[max(0, i - 2):i + 3], 0) for i in range(len(raw))])
ref = json.loads((U / 'ref' / f'{S}_umbilicus.json').read_text())
P = np.array(sorted([[p['z'], p['x'], p['y']] for p in ref['control_points']]), float)
hand = np.stack([np.interp(ZS, P[:, 0], P[:, 1]), np.interp(ZS, P[:, 0], P[:, 2])], 1)
out = {'hand': hand, 'unc': est(1.5), 'orig': est(0), 'shift': hand + np.array([300.0, 0.0])}
for name, xy in out.items():
    d = O / f'{PRE}_{name}'; d.mkdir(exist_ok=True)
    cps = [{'x': round(float(x), 1), 'y': round(float(y), 1), 'z': int(z)} for (x, y), z in zip(xy, ZS)]
    (d / 'umbilicus.json').write_text(json.dumps({'control_points': cps, 'provenance': {'variant': name, 'scroll': S, 'doc': __doc__}}, indent=1))
    print(PRE, name, 'median dist to hand (mm):', round(float(np.median(np.linalg.norm(xy - hand, axis=1)) * vox), 2), flush=True)
