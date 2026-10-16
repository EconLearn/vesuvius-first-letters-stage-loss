"""Automatic umbilicus vs Drobkov's manual umbilici (MIT) on 10 First Letters scrolls.

Baseline = the estimator VC3D's vc_gen_umbilicus uses (our port, tools/cloud/quiet/umb_auto.py): skeleton of the published m7
sheet prediction at level 2, in-plane normals from a 2D structure tensor (coherence > 0.5), RANSAC 1000 x 10000 on
sum cos^2(angle(p - c, n)) then a hill climb on sum cos^2 / max(100, dist).
Candidate = the same estimator after an affine "un-crush": sheet pixels are whitened by PCA with strength alpha
(x' = diag(lambda^-alpha/2) V^T (x - mu)), normals transform with the inverse transpose, the centre found in the whitened
frame is mapped back. Crushed scrolls have elliptical windings whose normals do not point at the axis; whitening makes
them closer to circular.
Error = distance in mm (full-res voxels x voxel size) to the manual axis interpolated at the same z.
Dev scrolls (choose alpha): PHerc0358, PHerc1203, PHerc0191. Test = the other 7, reported once with the chosen alpha.
usage: umb_bench.py DIR [--nz 8] [--alphas 0 0.5 1]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
import zarr

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--nz', type=int, default=8)
ap.add_argument('--alphas', type=float, nargs='*', default=[0.0, 0.5, 1.0]); ap.add_argument('--seeds', type=int, nargs='*', default=[1, 2, 3])
a = ap.parse_args(); D = Path(a.dir); C = D / 'cache'; C.mkdir(exist_ok=True)
SC = json.loads((D / 'scrolls.json').read_text()); DEV = ['PHerc0358', 'PHerc1203', 'PHerc0191']


def ref_at(s, z):
    j = json.loads((D / 'ref' / f'{s}_umbilicus.json').read_text())
    P = np.array(sorted([[p['z'], p['y'], p['x']] for p in j['control_points']]), float)
    return np.array([np.interp(z, P[:, 0], P[:, 1]), np.interp(z, P[:, 0], P[:, 2])])


def slice_l2(s, zl):
    f = C / f'{s}_{zl}.npz'
    if f.exists():
        z = np.load(f); return np.unpackbits(z['b'])[:int(np.prod(z['shape']))].reshape(z['shape']).astype(bool)
    for attempt in range(6):
        try:
            arr = zarr.open('s3://' + SC[s]['m7'] + '/2', mode='r', storage_options=dict(anon=True))
            sl = np.asarray(arr[zl]) > 0; break
        except Exception as e:
            print('  retry', s, zl, type(e).__name__, flush=True); time.sleep(10 * (attempt + 1))
    else:
        return None
    np.savez_compressed(f, b=np.packbits(sl), shape=sl.shape); return sl


def st2d(sheet, sd=1.0, si=2.0):
    f = sheet.astype(np.float32)
    gy = ndimage.gaussian_filter(f, sd, order=(1, 0)); gx = ndimage.gaussian_filter(f, sd, order=(0, 1))
    jyy = ndimage.gaussian_filter(gy * gy, si); jxx = ndimage.gaussian_filter(gx * gx, si); jyx = ndimage.gaussian_filter(gy * gx, si)
    tr = jyy + jxx; det = jyy * jxx - jyx * jyx; l1 = tr / 2 + np.sqrt(np.maximum(tr * tr / 4 - det, 0)); l2 = tr - l1
    vy = jyx; vx = l1 - jyy; alt = np.abs(vy) + np.abs(vx) < 1e-12
    vy = np.where(alt, l1 - jxx, vy); vx = np.where(alt, jyx, vx); n = np.hypot(vy, vx) + 1e-30
    return vy / n, vx / n, (l1 - l2) / (l1 + l2 + 1e-30)


def estimate(P, N, lo, hi, seed):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(P), 10000)
    P, N = P[idx].astype(np.float64), N[idx].astype(np.float64)
    cand = np.stack([rng.uniform(lo[0], hi[0], 1000), rng.uniform(lo[1], hi[1], 1000)], 1)
    best, bs = None, -1
    for c in cand:
        d = P - c; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12; s = ((d * N).sum(1) ** 2).sum()
        if s > bs: bs, best = s, c
    def score(c):
        d = P - c; dist = np.linalg.norm(d, axis=1); ok = dist > 1e-6; d = d[ok] / dist[ok, None]
        return (((d * N[ok]).sum(1) ** 2) / np.maximum(100.0, dist[ok])).sum()
    b = best.copy(); sb = score(b); step = 1024.0
    while step >= 1.0:
        moved = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    c = b + np.array([dx * step, dy * step]); s = score(c)
                    if s > sb: sb, b, moved = s, c, True
        if not moved: step /= 2
    return b


def centre(sheet, alpha, seeds):
    sk = skeletonize(sheet); vy, vx, coh = st2d(sheet); ok = sk & (coh > 0.5)
    yy, xx = np.nonzero(ok)
    if len(yy) < 1000: return None
    P = np.stack([yy * 4 + 1.5, xx * 4 + 1.5], 1); N = np.stack([vy[yy, xx], vx[yy, xx]], 1)
    sy, sx = np.nonzero(sheet); X = np.stack([sy * 4 + 1.5, sx * 4 + 1.5], 1).astype(np.float64); mu = X.mean(0)
    lam, V = np.linalg.eigh(np.cov((X - mu).T)); s0 = np.sqrt(lam.mean())
    W = np.diag((lam / lam.mean()) ** (-alpha / 2)) @ V.T                    # unit-preserving at alpha = 0 (W = V^T)
    Pw = (P - mu) @ W.T; Nw = N @ np.linalg.inv(W); Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
    lo, hi = Pw.min(0), Pw.max(0)
    ests = np.array([estimate(Pw, Nw, lo, hi, sd) for sd in seeds]); cw = np.median(ests, 0)
    return np.linalg.solve(W, cw) + mu, float(np.ptp(ests, 0).max()), float(np.sqrt(lam.max() / lam.min()))


res = {}
for s, m in SC.items():
    z0, z1 = m['z_range']; zs = np.linspace(z0 + 0.1 * (z1 - z0), z1 - 0.1 * (z1 - z0), a.nz).round().astype(int)
    for z in zs:
        t = time.time(); zl = int(z // 4)
        if zl >= m['l2_shape'][0]: continue
        sheet = slice_l2(s, zl); ref = ref_at(s, z)
        if sheet is None: continue
        for al in a.alphas:
            r = centre(sheet, al, a.seeds)
            if r is None: continue
            c, spread, aniso = r
            err_mm = float(np.linalg.norm(c - ref) * m['vox_um'] / 1000)
            res.setdefault(s, {}).setdefault(str(al), []).append(dict(z=int(z), err_mm=round(err_mm, 3), spread_vx=round(spread, 1), aniso=round(aniso, 2), c=[float(c[0]), float(c[1])], ref=[float(ref[0]), float(ref[1])]))
        print(s, z, 'aniso %.2f' % aniso, ' '.join(f"a{al}:{res[s][str(al)][-1]['err_mm']:.2f}mm" for al in a.alphas if str(al) in res.get(s, {})), f'{time.time() - t:.0f}s', flush=True)
(D / 'bench_raw.json').write_text(json.dumps(res, indent=1))

def theil_sen(z, v):
    sl = [(v[j] - v[i]) / (z[j] - z[i]) for i in range(len(z)) for j in range(i + 1, len(z)) if z[j] != z[i]]
    m = np.median(sl) if sl else 0.0; b = np.median(np.array(v) - m * np.array(z)); return m, b

def errs(s, al, smooth):
    rr = res[s][str(al)]
    if not smooth: return [r['err_mm'] for r in rr]
    z = np.array([r['z'] for r in rr], float); cy = [r['c'][0] for r in rr]; cx = [r['c'][1] for r in rr]
    my, by = theil_sen(z, cy); mx, bx = theil_sen(z, cx)
    return [float(np.hypot(my * r['z'] + by - r['ref'][0], mx * r['z'] + bx - r['ref'][1]) * SC[s]['vox_um'] / 1000) for r in rr]

print()
print('per-scroll median error (mm): raw | Theil-Sen line across z')
for s in SC:
    if s not in res: continue
    print(s, ' '.join(f"a{al}={np.median(errs(s, al, False)):.2f}|{np.median(errs(s, al, True)):.2f}" for al in a.alphas), '| dev' if s in DEV else '| test')
for grp in ('dev', 'test'):
    ss = [s for s in res if (s in DEV) == (grp == 'dev')]
    if not ss: continue
    for sm in (False, True):
        print(grp, 'smoothed' if sm else 'raw', ' '.join(f"a{al}: {np.median([np.median(errs(s, al, sm)) for s in ss]):.2f} mm" for al in a.alphas))
