"""Controls for the un-crush result, cached slices only (test scrolls):
  orig    = the exact original estimator (umb_auto.py: candidates uniform over the full image extent, no rotation)
  base    = my alpha 0 baseline (rotated to PCA axes, candidates over the points' bbox) - must match orig
  uncrush = alpha 0.5 on the PCA axes (the chosen method)
  sabot   = alpha 0.5 anisotropic scaling along a RANDOM axis pair (seeded per slice) - must NOT match uncrush's gain
usage: umb_ctrl.py DIR"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.argv = [sys.argv[0], sys.argv[1], '--nz', '8']
D = Path(sys.argv[1])
src = (Path(__file__).parent / 'umb_bench.py').read_text()
ns = {}; exec(src.split("res = {}")[0].replace("a = ap.parse_args()", "a = ap.parse_args(sys.argv[1:])"), ns)   # functions only
st2d, estimate, skeletonize, ref_at, SC = ns['st2d'], ns['estimate'], ns['skeletonize'], ns['ref_at'], ns['SC']
DEV = {'PHerc0358', 'PHerc1203', 'PHerc0191'}
def load(s, zl):
    f = D / 'cache' / f'{s}_{zl}.npz'
    if not f.exists(): return None
    z = np.load(f); return np.unpackbits(z['b'])[:int(np.prod(z['shape']))].reshape(z['shape']).astype(bool)
def run(sheet, mode, rng):
    sk = skeletonize(sheet); vy, vx, coh = st2d(sheet); ok = sk & (coh > 0.5); yy, xx = np.nonzero(ok)
    P = np.stack([yy * 4 + 1.5, xx * 4 + 1.5], 1); N = np.stack([vy[yy, xx], vx[yy, xx]], 1)
    if mode == 'orig':
        Y, X = sheet.shape; ests = np.array([estimate(P, N, (0, 0), (Y * 4, X * 4), sd) for sd in (1, 2, 3)]); return np.median(ests, 0)
    sy, sx = np.nonzero(sheet); Xs = np.stack([sy * 4 + 1.5, sx * 4 + 1.5], 1).astype(np.float64); mu = Xs.mean(0)
    lam, V = np.linalg.eigh(np.cov((Xs - mu).T))
    if mode == 'sabot':
        th = rng.uniform(0, np.pi); V = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    al = 0.0 if mode == 'base' else 0.5
    W = np.diag((lam / lam.mean()) ** (-al / 2)) @ V.T
    Pw = (P - mu) @ W.T; Nw = N @ np.linalg.inv(W); Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
    ests = np.array([estimate(Pw, Nw, Pw.min(0), Pw.max(0), sd) for sd in (1, 2, 3)]); return np.linalg.solve(W, np.median(ests, 0)) + mu
raw = json.loads((D / 'bench_raw.json').read_text()); out = {m: [] for m in ('orig', 'base', 'uncrush', 'sabot')}
for s, d in raw.items():
    if s in DEV: continue
    for i, rec in enumerate(d['0.0']):
        z = rec['z']; sheet = load(s, z // 4)
        if sheet is None: continue
        ref = ref_at(s, z); rng = np.random.default_rng(1000 * i + len(s))
        for m in out:
            c = run(sheet, m, rng); out[m].append(float(np.linalg.norm(c - ref) * SC[s]['vox_um'] / 1000))
        print(s, z, ' '.join(f'{m}={out[m][-1]:.2f}' for m in out), flush=True)
A = {m: np.array(v) for m, v in out.items()}
for m in A: print(f'{m:8s} n={len(A[m])} median {np.median(A[m]):.2f} mm  within 1 mm {np.mean(A[m] < 1):.0%}  within 2 mm {np.mean(A[m] < 2):.0%}')
for m in ('base', 'uncrush', 'sabot'):
    print(f'{m} vs orig: better {np.sum(A[m] < A["orig"] - 0.05)} worse {np.sum(A[m] > A["orig"] + 0.05)}  Wilcoxon(less) p={wilcoxon(A[m], A["orig"], alternative="less").pvalue:.4f}')
print(f'uncrush vs sabot: Wilcoxon(less) p={wilcoxon(A["uncrush"], A["sabot"], alternative="less").pvalue:.4f}')
(D / 'controls.json').write_text(json.dumps(out, indent=1))
