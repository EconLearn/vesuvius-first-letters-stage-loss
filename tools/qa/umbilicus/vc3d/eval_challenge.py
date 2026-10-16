"""Pre-registered 2026-10-15: the patched C++ estimator, uncrush 0 vs 1.5, on the challenge's own manual umbilici
(PHerc0125, 0211, 0826 at L0; 0332, 1451 with level-2 grids; Paris4), 8 slices per scroll at 10..90 % of the annotated z
range, seeds 1..9. Measured: midpoint version (alpha 0.5) 0.85 -> 0.69 mm, p = 0.0725; normal version (1.5) 0.85 -> 0.61 mm, 20 better /
19 worse, p = 0.113. The pre-registered pass (p < 0.05) was NOT met by either.
usage: list_challenge.py DIR; meta_challenge.py DIR; eval_challenge.py DIR HARNESS"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, s3fs
from scipy.stats import wilcoxon
D = Path(sys.argv[1]); G = D / 'grids'; G.mkdir(exist_ok=True)
L = json.loads((D / 'listing.json').read_text()); fs = s3fs.S3FileSystem(anon=True); H = sys.argv[2]; SEEDS = range(1, 10)
CFG = {'PHerc0125': 1, 'PHerc0211': 1, 'PHerc0826': 1, 'PHerc0332': 4, 'PHerc1451': 4, 'PHercParis4': 1}   # grid level factor
out = {'0': {}, '1.5': {}}; zsl = {}
for s, f in CFG.items():
    u = json.loads((D / f'{s}_umbilicus.json').read_text()); md = u['metadata']; vox = md['voxelsize_um'] * f / 1000
    P = np.array(sorted([[p['z'], p['x'], p['y']] for p in u['control_points']]), float)
    z0, z1 = P[0, 0], P[-1, 0]; zs = [int(round((z0 + q * (z1 - z0)) / f)) for q in np.linspace(0.1, 0.9, 8)]
    files, refs = [], []
    for zg in zs:
        fp = G / f'{s}_{zg:06d}.grid'
        if not fp.exists():
            for att in range(5):
                try: fp.write_bytes(fs.cat_file(f"{L[s]['ng'][0]}/xy/{zg:06d}.grid")); break
                except Exception: time.sleep(5 * (att + 1))
        files.append(str(fp)); zf = zg * f
        refs.append(np.array([np.interp(zf, P[:, 0], P[:, 1]), np.interp(zf, P[:, 0], P[:, 2])]) / f)
    zsl[s] = zs
    for a in out:
        est = []
        for sd in SEEDS:
            r = subprocess.run([H, a, str(sd)] + files, capture_output=True, text=True).stdout.split('\n')
            est.append([list(map(float, l.split())) for l in r if l.strip()])
        med = np.median(np.array(est), 0)
        out[a][s] = [float(np.linalg.norm(med[i] - refs[i]) * vox) for i in range(len(zs))]
    print(s, 'sizes MB', round(sum(Path(x).stat().st_size for x in files) / 1e6, 1), 'orig', np.round(out['0'][s], 2).tolist(), '| uncrush', np.round(out['1.5'][s], 2).tolist(), flush=True)
A = {a: np.concatenate([out[a][s] for s in CFG]) for a in out}
p = wilcoxon(A['1.5'], A['0'], alternative='less').pvalue
print(f"ALL n={len(A['0'])}: orig median {np.median(A['0']):.2f} mm <1mm {np.mean(A['0'] < 1):.0%} <2mm {np.mean(A['0'] < 2):.0%} | uncrush median {np.median(A['1.5']):.2f} <1mm {np.mean(A['1.5'] < 1):.0%} <2mm {np.mean(A['1.5'] < 2):.0%} | "
      f"better {np.sum(A['1.5'] < A['0'] - 0.05)} worse {np.sum(A['1.5'] > A['0'] + 0.05)} p={p:.4f} -> {'PASS' if p < 0.05 and np.median(A['1.5']) < np.median(A['0']) else 'FAIL'}")
print('per-scroll median orig -> uncrush: ' + ' '.join(f"{s[5:]}:{np.median(out['0'][s]):.2f}->{np.median(out['1.5'][s]):.2f}" for s in CFG))
(D / 'chal_results.json').write_text(json.dumps({'z': zsl, 'err_mm': out, 'p': p}, indent=1))
