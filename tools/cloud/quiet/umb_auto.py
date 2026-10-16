"""Automatic umbilicus for a scroll without a published one. Same estimator as the port of VC3D
core/src/normalgridtools.cpp::align_and_extract_umbilicus (the function vc_gen_umbilicus runs):
  - 10,000 sampled sheet points with in-plane sheet normals,
  - RANSAC: 1,000 uniform candidates, score = sum cos^2(angle(point - c, normal)),
  - hill climb from step 1024 down to 1 on score = sum cos^2 / max(100, dist).
Sheet points = skeleton of the published m7 surface prediction at level 2 (label-free).
In-plane normals, selectable:
  --normals m7st     2D structure tensor of the same m7 L2 slice (sigma_d 1, sigma_i 2 L2 px), coherence > 0.5
                     (needs nothing but m7; the route for scrolls without a published lasagna)
  --normals lasagna  published lasagna nx/ny at level 2 (only PHerc0343 / PHerc1447 have them)
Coordinates: full-res voxels (level-2 index * 4 + 1.5, pixel centre).  NO organiser surface or label is read.

usage: umb_auto.py SCROLL OUT.json Z0 Z1 STEP [--normals m7st|lasagna] [--las DIR] [--seeds 1 2 3]
"""
import os, sys, json, time, argparse
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7normals as M

TMP = M.TMP


def estimate(pts, nrm, extent, seed, lo=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pts), 10000)
    P, N = pts[idx].astype(np.float64), nrm[idx].astype(np.float64)
    cand = np.stack([rng.uniform(lo[0], extent[0], 1000), rng.uniform(lo[1], extent[1], 1000)], 1)
    best, bs = None, -1
    for c in cand:
        d = P - c; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        s = ((d * N).sum(1) ** 2).sum()
        if s > bs:
            bs, best = s, c

    def score(c):
        d = P - c; dist = np.linalg.norm(d, axis=1); ok = dist > 1e-6
        d = d[ok] / dist[ok, None]
        return (((d * N[ok]).sum(1) ** 2) / np.maximum(100.0, dist[ok])).sum()
    b = best.copy(); sb = score(b); step = 1024.0
    while step >= 1.0:
        moved = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                c = b + np.array([dx * step, dy * step]); s = score(c)
                if s > sb:
                    sb, b, moved = s, c, True
        if not moved:
            step /= 2
    return b, sb


def st2d(sheet, sd=1.0, si=2.0):
    f = sheet.astype(np.float32)
    gy = ndimage.gaussian_filter(f, sd, order=(1, 0)); gx = ndimage.gaussian_filter(f, sd, order=(0, 1))
    jyy = ndimage.gaussian_filter(gy * gy, si); jxx = ndimage.gaussian_filter(gx * gx, si)
    jyx = ndimage.gaussian_filter(gy * gx, si)
    # principal eigenvector of [[jyy, jyx], [jyx, jxx]]
    tr = jyy + jxx; det = jyy * jxx - jyx * jyx
    l1 = tr / 2 + np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l2 = tr - l1
    vy = jyx; vx = l1 - jyy
    alt = np.abs(vy) + np.abs(vx) < 1e-12
    vy = np.where(alt, l1 - jxx, vy); vx = np.where(alt, jyx, vx)
    n = np.hypot(vy, vx) + 1e-30
    coh = (l1 - l2) / (l1 + l2 + 1e-30)
    return vy / n, vx / n, coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll'); ap.add_argument('out'); ap.add_argument('z0', type=int); ap.add_argument('z1', type=int)
    ap.add_argument('step', type=int)
    ap.add_argument('--normals', default='m7st', choices=['m7st', 'lasagna'])
    ap.add_argument('--las', default=None, help='dir holding *_nx.ome.zarr / *_ny.ome.zarr (level 2)')
    ap.add_argument('--seeds', type=int, nargs='*', default=[1, 2, 3])
    ap.add_argument('--keep-cache', action='store_true')
    ap.add_argument('--uncrush', type=float, default=0.0,
                    help='PCA whitening strength alpha of each slice before the estimate (0 = VC3D estimator unchanged); '
                         'crushed scrolls have elliptical windings whose normals do not point at the axis')
    a = ap.parse_args()
    try:
        os.nice(19 - os.nice(0))
    except Exception:
        pass
    s = a.scroll
    cache = f'{TMP}/m7cache_{s}/2'
    m7 = M.ChunkMirror(M.M7_TMPL.format(s=s, v=M.VOL[s]) + '/2', cache, workers=16)
    if a.normals == 'lasagna':
        import zarr, glob
        nxa = zarr.open(glob.glob(f'{a.las}/*_nx.ome.zarr')[0] + '/2', mode='r')
        nya = zarr.open(glob.glob(f'{a.las}/*_ny.ome.zarr')[0] + '/2', mode='r')
    res = []
    Y, X = m7.shape[1], m7.shape[2]
    for z in range(a.z0, a.z1 + 1, a.step):
        t = time.time(); zl = z // 4
        sheet = m7.read_box(zl, zl + 1, 0, Y, 0, X)[0] > 0
        sk = skeletonize(sheet)
        if a.normals == 'lasagna':
            nx = np.asarray(nxa[zl]).astype(np.float32); ny = np.asarray(nya[zl]).astype(np.float32)
            valid = (nx != 0) | (ny != 0)
            vx = (nx - 128) / 127; vy = (ny - 128) / 127
            nn = np.sqrt(vx ** 2 + vy ** 2)
            ok = sk & valid & (nn > 0.3)
            yy, xx = np.nonzero(ok)
            N = np.stack([vy[yy, xx], vx[yy, xx]], 1) / nn[yy, xx, None]
        else:
            vy, vx, coh = st2d(sheet)
            ok = sk & (coh > 0.5)
            yy, xx = np.nonzero(ok)
            N = np.stack([vy[yy, xx], vx[yy, xx]], 1)
        if len(yy) < 1000:
            print(json.dumps(dict(z=z, skipped='too few sheet points', n=int(len(yy)))), flush=True)
            continue
        P = np.stack([yy * 4 + 1.5, xx * 4 + 1.5], 1)
        ests = []
        if a.uncrush > 0:   # estimate in an un-crushed frame: x' = diag(lambda^-alpha/2) V^T (x - mu); normals by W^-T
            sy, sx = np.nonzero(sheet)
            Xs = np.stack([sy * 4 + 1.5, sx * 4 + 1.5], 1).astype(np.float64); mu = Xs.mean(0)
            lam, V = np.linalg.eigh(np.cov((Xs - mu).T))
            W = np.diag((lam / lam.mean()) ** (-a.uncrush / 2)) @ V.T
            Pw = (P - mu) @ W.T; Nw = N @ np.linalg.inv(W); Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
            for sd in a.seeds:
                cw, sc = estimate(Pw, Nw, tuple(Pw.max(0)), sd, lo=tuple(Pw.min(0)))
                c = np.linalg.solve(W, cw) + mu
                ests.append([float(c[0]), float(c[1]), float(sc)])
        else:
            for sd in a.seeds:
                c, sc = estimate(P, N, (Y * 4, X * 4), sd)
                ests.append([float(c[0]), float(c[1]), float(sc)])
        E = np.array(ests)
        rec = dict(z=z, n_sheet_px=int(sheet.sum()), n_skel=int(len(P)), seeds=a.seeds, yx_per_seed=E[:, :2].tolist(),
                   yx_median=np.median(E[:, :2], 0).tolist(), spread_vox=float(np.ptp(E[:, :2], 0).max()),
                   secs=round(time.time() - t, 1))
        print(json.dumps(rec), flush=True)
        res.append(rec)
    cps = [dict(x=round(r['yx_median'][1], 1), y=round(r['yx_median'][0], 1), z=r['z'], score=100) for r in res]
    json.dump(dict(control_points=cps, provenance=dict(
        method='port of VC3D align_and_extract_umbilicus (RANSAC 1000 x 10000 + weighted hill climb)' + (f'; un-crush PCA whitening alpha {a.uncrush}' if a.uncrush > 0 else ''),
        inputs=f'{s} m7 L2 skeleton + in-plane normals from {a.normals} (published, label-free)',
        per_slice=res)), open(a.out, 'w'), indent=1)
    print('wrote', a.out)
    if not a.keep_cache:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == '__main__':
    main()
