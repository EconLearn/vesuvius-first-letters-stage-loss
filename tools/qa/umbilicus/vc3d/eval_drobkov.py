"""The PATCHED C++ function (extracted verbatim from normalgridtools.cpp by build.sh, compiled against OpenCV stand-ins) on
the 80 published normal-grid slices of umb_vc3d.py (8 per scroll, cached in DIR/grids), vs Drobkov's manual umbilici.
uncrush 0 (unchanged; the whiten = false path is the original code draw for draw) and 1.5 (normal un-crush; beta
chosen on the 3 dev scrolls 0358/1203/0191); seeds 1..9, median per slice. Paired one-sided Wilcoxon, held-out 7 and all 10.
Measured 2026-10-15 on the 7 held-out scrolls: 2.22 -> 1.33 mm, 35 better / 18 worse, p = 0.0074.
usage: eval_drobkov.py DIR HARNESS"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
U = Path(sys.argv[1]); H = sys.argv[2]; SC = json.loads((U / 'scrolls.json').read_text()); raw = json.loads((U / 'bench_raw.json').read_text())
SEEDS = tuple(range(1, 10)); DEV = {'PHerc0358', 'PHerc1203', 'PHerc0191'}
def ref_at(s, z):
    j = json.loads((U / 'ref' / f'{s}_umbilicus.json').read_text())
    P = np.array(sorted([[p['z'], p['x'], p['y']] for p in j['control_points']]), float)
    return np.array([np.interp(z, P[:, 0], P[:, 1]), np.interp(z, P[:, 0], P[:, 2])])
out = {'0': {}, '1.5': {}}
for s in SC:
    if s not in raw: continue
    vox = SC[s]['vox_um'] / 1000
    files = [str(U / 'grids' / f"{s}_{rec['z']:06d}.grid") for rec in raw[s]['0.0']]
    zs = [rec['z'] for rec in raw[s]['0.0']]
    for a in out:
        est = np.array([[list(map(float, l.split())) for l in subprocess.run([H, a, str(sd)] + files, capture_output=True, text=True).stdout.split('\n') if l.strip()] for sd in SEEDS])
        med = np.median(est, 0)
        out[a][s] = [float(np.linalg.norm(med[i] - ref_at(s, z)) * vox) for i, z in enumerate(zs)]
    print(s, 'orig', np.round(np.median(out['0'][s]), 2), 'uncrush', np.round(np.median(out['1.5'][s]), 2), flush=True)
py = json.loads((U / 'vc3d_results.json').read_text())
for grp, ss in (('test (7)', [s for s in out['0'] if s not in DEV]), ('all (10)', list(out['0']))):
    A = {a: np.concatenate([out[a][s] for s in ss]) for a in out}
    B = {m: np.concatenate([py[m][s] for s in ss]) for m in ('orig', 'uncrush')}
    print(f"{grp} n={len(A['0'])}: C++ orig median {np.median(A['0']):.2f} <1mm {np.mean(A['0'] < 1):.0%} | C++ uncrush {np.median(A['1.5']):.2f} <1mm {np.mean(A['1.5'] < 1):.0%} | "
          f"better {np.sum(A['1.5'] < A['0'] - 0.05)} worse {np.sum(A['1.5'] > A['0'] + 0.05)} p={wilcoxon(A['1.5'], A['0'], alternative='less').pvalue:.4f} || "
          f"python port: orig {np.median(B['orig']):.2f} uncrush {np.median(B['uncrush']):.2f}")
(U / 'vc3d_cpp_results.json').write_text(json.dumps(out, indent=1))
