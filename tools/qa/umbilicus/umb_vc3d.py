"""Un-crush on VC3D's OWN input: the published normal grids (GridStore v3 'VCGS', xy/<z>.grid, full-res coordinates).
VC3D's align_and_extract_umbilicus samples 10,000 path segments (midpoint + perpendicular normal), scores 1,000 uniform
candidates over the grid bounds with sum cos^2, then hill-climbs on sum cos^2 / max(100, dist). Here: that estimator (orig),
the same after PCA un-crush of the segment midpoints with alpha 0.5 (FIXED in advance - chosen on the m7 route's dev
scrolls, no tuning here), and a random-axis sabotage of the same strength. Truth: Drobkov's manual umbilici. Same z as
umb_bench (bench_raw.json). Paired one-sided Wilcoxon on the 7 scrolls that were TEST scrolls in the m7 benchmark, plus all 10.
usage: umb_vc3d.py DIR"""
import json, struct, sys, time
from pathlib import Path
import numpy as np, s3fs
from scipy.stats import wilcoxon

D = Path(sys.argv[1]); SC = json.loads((D / 'scrolls.json').read_text()); raw = json.loads((D / 'bench_raw.json').read_text())
DEV = {'PHerc0358', 'PHerc1203', 'PHerc0191'}; fs = s3fs.S3FileSystem(anon=True); G = D / 'grids'; G.mkdir(exist_ok=True)


def ref_at(s, z):
    j = json.loads((D / 'ref' / f'{s}_umbilicus.json').read_text())
    P = np.array(sorted([[p['z'], p['y'], p['x']] for p in j['control_points']]), float)
    return np.array([np.interp(z, P[:, 0], P[:, 1]), np.interp(z, P[:, 0], P[:, 2])])


def read_grid(s, z):
    f = G / f'{s}_{z:06d}.grid'
    if not f.exists():
        ng = SC[s]['m7'].replace('.zarr', '.normal-grids')
        for att in range(5):
            try:
                f.write_bytes(fs.cat_file(f'{ng}/xy/{z:06d}.grid')); break
            except Exception:
                time.sleep(5 * (att + 1))
        else:
            return None
    b = f.read_bytes()
    if len(b) < 52 or b[:4] != b'VCGS': return None
    h = struct.unpack('>13I', b[:52]); bw, bh, npaths, poff, joff = h[4], h[5], h[8], h[10], h[11]
    segs_p, segs_t = [], []; o = poff
    for _ in range(npaths):
        x0, y0, n = struct.unpack('>3I', b[o:o + 12]); o += 12
        d = np.frombuffer(b[o:o + n], dtype=np.int8).astype(np.int64).reshape(-1, 2); o += n
        pts = np.vstack([[x0, y0], np.array([x0, y0]) + np.cumsum(d, 0)]) if len(d) else np.array([[x0, y0]])
        if len(pts) < 2: continue
        p1, p2 = pts[:-1], pts[1:]; segs_p.append((p1 + p2) / 2.0); segs_t.append(p2 - p1)
    if not segs_p: return None
    P = np.vstack(segs_p); T = np.vstack(segs_t).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    N = np.stack([-T[:, 1], T[:, 0]], 1)                                    # perpendicular, as VC3D
    return P[:, ::-1].copy(), N[:, ::-1].copy(), (bh, bw)                   # (y, x) order


def estimate(P, N, lo, hi, seed):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(P), 10000); P, N = P[idx], N[idx]
    cand = np.stack([rng.uniform(lo[0], hi[0], 1000), rng.uniform(lo[1], hi[1], 1000)], 1); best, bs = None, -1
    for c in cand:
        d = P - c; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12; sc = ((d * N).sum(1) ** 2).sum()
        if sc > bs: bs, best = sc, c
    def score(c):
        d = P - c; dist = np.linalg.norm(d, axis=1); ok = dist > 1e-6; d = d[ok] / dist[ok, None]
        return (((d * N[ok]).sum(1) ** 2) / np.maximum(100.0, dist[ok])).sum()
    b = best.copy(); sb = score(b); step = 1024.0
    while step >= 1.0:
        moved = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    c = b + np.array([dx * step, dy * step]); sc = score(c)
                    if sc > sb: sb, b, moved = sc, c, True
        if not moved: step /= 2
    return b


def run(P, N, shape, mode, rng):
    if mode == 'orig':
        return np.median([estimate(P, N, (0, 0), shape, sd) for sd in (1, 2, 3)], 0)
    mu = P.mean(0); lam, V = np.linalg.eigh(np.cov((P - mu).T))
    if mode == 'sabot':
        th = rng.uniform(0, np.pi); V = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    W = np.diag((lam / lam.mean()) ** (-0.5 / 2)) @ V.T
    Pw = (P - mu) @ W.T; Nw = N @ np.linalg.inv(W); Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
    cw = np.median([estimate(Pw, Nw, Pw.min(0), Pw.max(0), sd) for sd in (1, 2, 3)], 0)
    return np.linalg.solve(W, cw) + mu


out = {m: {} for m in ('orig', 'uncrush', 'sabot')}
for s in SC:
    if s not in raw: continue
    vox = SC[s]['vox_um'] / 1000
    for i, rec in enumerate(raw[s]['0.0']):
        z = rec['z']; g = read_grid(s, z)
        if g is None: print(s, z, 'no grid'); continue
        P, N, shape = g; ref = ref_at(s, z); rng = np.random.default_rng(1000 * i + len(s))
        for m in out:
            out[m].setdefault(s, []).append(float(np.linalg.norm(run(P, N, shape, m, rng) - ref) * vox))
        print(s, z, f'{len(P)} segs', ' '.join(f'{m}={out[m][s][-1]:.2f}' for m in out), flush=True)
(D / 'vc3d_results.json').write_text(json.dumps(out, indent=1))
for grp, ss in (('test (7)', [s for s in out['orig'] if s not in DEV]), ('all (10)', list(out['orig']))):
    A = {m: np.concatenate([out[m][s] for s in ss]) for m in out}
    print(f'{grp}: n={len(A["orig"])} slices | ' + ' | '.join(f'{m} median {np.median(A[m]):.2f} mm, <1 mm {np.mean(A[m] < 1):.0%}, <2 mm {np.mean(A[m] < 2):.0%}' for m in A))
    print(f'   uncrush vs orig: better {np.sum(A["uncrush"] < A["orig"] - 0.05)} worse {np.sum(A["uncrush"] > A["orig"] + 0.05)} '
          f'p={wilcoxon(A["uncrush"], A["orig"], alternative="less").pvalue:.4f}; sabot vs orig p={wilcoxon(A["sabot"], A["orig"], alternative="less").pvalue:.4f}; '
          f'uncrush vs sabot p={wilcoxon(A["uncrush"], A["sabot"], alternative="less").pvalue:.4f}')
    print('   per-scroll median orig -> uncrush: ' + ' '.join(f"{s[5:]}:{np.median(out['orig'][s]):.2f}->{np.median(out['uncrush'][s]):.2f}" for s in ss))
