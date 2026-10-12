#!/usr/bin/env python3
# Copied unchanged (below the licence) from github.com/Nieuwlaar/pherc343-first-letters, commit e8370f1,
# mesh/snap_meshes.py. Its licence:
#
# MIT License
#
# Copyright (c) 2026 Erwin Nieuwlaar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""snap_meshes.py <meshes_dir> <scroll> <out_dir> --windings windings.txt [--z0 Z0 --z1 Z1]
       [--max-off 15] [--jump 6] [--sigma 1.0] [--level 0] [--procs 8]
       (environment: SURF_ZARR = the m7 surface prediction of the scroll, a zarr with resolution levels)

Post-hoc snap of fitted spiral windings onto the surface prediction. For every vertex of each listed tifxyz winding
(grid step 20 vox), move along the local mesh normal to the centre of the NEAREST predicted-sheet run within
+/- max-off voxels (binary prediction -> run centre with a parabolic sub-voxel refinement on a [1,2,1]-smoothed
profile). The offset field is then regularised so the mesh cannot hop between neighbouring sheets: offsets deviating
> --jump vox from the 5x5 nan-median of their neighbourhood are rejected, rejected/missing offsets are filled by
normalised convolution from their neighbours, and the field is smoothed with a Gaussian of --sigma grid cells.
Vertices with no sheet within range keep their fitted position.
Writes <out_dir>/<winding>/{x,y,z}.tif + meta.json (bbox recomputed) and prints a per-winding report (fraction
snapped, median |offset|, rejected fraction); <out_dir>/snap_report.tsv.
The meshes in outputs/ were snapped with the defaults above, at level 0, between fit_spiral and lasagna flattening.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import tifffile, zarr
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('meshes'); ap.add_argument('scroll'); ap.add_argument('out')
ap.add_argument('--windings', required=True)
ap.add_argument('--z0', type=float, default=None); ap.add_argument('--z1', type=float, default=None)
ap.add_argument('--max-off', type=int, default=15); ap.add_argument('--jump', type=float, default=6.0)
ap.add_argument('--sigma', type=float, default=1.0); ap.add_argument('--level', type=int, default=0)
ap.add_argument('--procs', type=int, default=int(os.environ.get('RANK_PROCS', '8')))
a = ap.parse_args()
t0 = time.time()
SURF = os.environ.get('SURF_ZARR') or sys.exit('set SURF_ZARR to the surface prediction zarr')
surf = zarr.open(SURF, mode='r')[str(a.level)]
ds = 2 ** a.level; cz, cy, cx = surf.chunks; Z, Y, X = surf.shape
K = a.max_off; offs = np.arange(-K, K + 1, dtype=np.float32)   # 2K+1 samples along the normal
wlist = [l.strip() for l in open(a.windings) if l.strip() and not l.startswith('#')]
os.makedirs(a.out, exist_ok=True)

def load(d):
    xs = tifffile.imread(os.path.join(d, 'x.tif')).astype(np.float32)
    ys = tifffile.imread(os.path.join(d, 'y.tif')).astype(np.float32)
    zs = tifffile.imread(os.path.join(d, 'z.tif')).astype(np.float32)
    valid = (xs > -0.5) & (ys > -0.5) & (zs > -0.5) & np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    return xs, ys, zs, valid

def normals(xs, ys, zs, valid):
    P = np.stack([xs, ys, zs], -1).astype(np.float64); P[~valid] = np.nan
    # central differences where both neighbours valid, else one-sided (nan-aware)
    def grad(axis):
        g = np.full_like(P, np.nan)
        fwd = np.roll(P, -1, axis) - P; bwd = P - np.roll(P, 1, axis)
        n = P.shape[axis]
        idx = [slice(None)] * 3
        # invalidate wrap-around
        idx[axis] = slice(n - 1, n); fwd[tuple(idx)] = np.nan
        idx[axis] = slice(0, 1); bwd[tuple(idx)] = np.nan
        both = np.isfinite(fwd).all(-1) & np.isfinite(bwd).all(-1)
        g[both] = 0.5 * (fwd[both] + bwd[both])
        only_f = np.isfinite(fwd).all(-1) & ~both; g[only_f] = fwd[only_f]
        only_b = np.isfinite(bwd).all(-1) & ~both; g[only_b] = bwd[only_b]
        return g
    tu = grad(1); tv = grad(0)
    n = np.cross(tu, tv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(nn > 1e-6, nn, np.nan)
    ok = np.isfinite(n).all(-1) & valid
    return n.astype(np.float32), ok

# ---- gather all sample points of all windings ---------------------------------
meshes = {}; pts = []; owner = []
for w in wlist:
    d = os.path.join(a.meshes, w)
    if not os.path.exists(os.path.join(d, 'x.tif')):
        print(f'{w}: missing under {a.meshes}', file=sys.stderr); continue
    xs, ys, zs, valid = load(d)
    n, ok = normals(xs, ys, zs, valid)
    meshes[w] = dict(xs=xs, ys=ys, zs=zs, valid=valid, n=n, ok=ok, dir=d)
    vy, vx = np.nonzero(ok)
    base = np.stack([zs[vy, vx], ys[vy, vx], xs[vy, vx]], -1)           # zyx
    nv = n[vy, vx][:, ::-1]                                               # normal as zyx
    S = base[:, None, :] + offs[None, :, None] * nv[:, None, :]          # (V, 2K+1, 3)
    pts.append(S.reshape(-1, 3) / ds); owner.append(np.full(S.shape[0] * S.shape[1], len(meshes) - 1, np.int32))
    meshes[w]['vy'] = vy; meshes[w]['vx'] = vx
if not meshes: sys.exit('no meshes')
pts = np.concatenate(pts).astype(np.float32); owner = np.concatenate(owner)
print(f'{len(meshes)} windings, {len(pts)/1e6:.1f} M sample points; load {time.time()-t0:.0f}s', flush=True)
vals = np.zeros(len(pts), np.uint8)
inb = (pts[:, 0] >= 0) & (pts[:, 0] < Z - 0.5) & (pts[:, 1] >= 0) & (pts[:, 1] < Y - 0.5) & (pts[:, 2] >= 0) & (pts[:, 2] < X - 0.5)
pi = np.rint(pts).astype(np.int64); pi[:, 0] = np.clip(pi[:, 0], 0, Z - 1); pi[:, 1] = np.clip(pi[:, 1], 0, Y - 1); pi[:, 2] = np.clip(pi[:, 2], 0, X - 1)
ci = np.stack([pi[:, 0] // cz, pi[:, 1] // cy, pi[:, 2] // cx], -1)
key = ci[:, 0] * 100000 + ci[:, 1] * 300 + ci[:, 2]
order = np.argsort(key, kind='stable'); key_s = key[order]
uniq, starts = np.unique(key_s, return_index=True); ends = np.append(starts[1:], len(key_s))

def work(chunk_ids):
    sf = zarr.open(SURF, mode='r')[str(a.level)]
    out = []
    for nn_ in chunk_ids:
        s, e = starts[nn_], ends[nn_]; sel = order[s:e]; c = ci[sel[0]]
        z0 = int(c[0] * cz); y0 = int(c[1] * cy); x0 = int(c[2] * cx)
        blk = np.asarray(sf[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx])
        if not blk.any(): continue
        p = pi[sel]
        v = blk[p[:, 0] - z0, p[:, 1] - y0, p[:, 2] - x0]
        h = v > 0
        out.append((sel[h], v[h]))
    return out

import multiprocessing as mp
NP = max(1, a.procs)
groups = [list(range(i, len(uniq), NP)) for i in range(NP)]
with mp.get_context('fork').Pool(NP) as pool:
    for res in pool.imap_unordered(work, groups):
        for idx, v in res: vals[idx] = v
vals[~inb] = 0
print(f'sampled {len(uniq)} chunks in {time.time()-t0:.0f}s ({NP} procs)', flush=True)

# ---- per winding: choose nearest run, regularise, apply -------------------------
def nanmedian_filter(d, size=5):
    # nan-aware median via generic_filter is slow; use a stack of shifted copies
    H, W = d.shape; r = size // 2
    pad = np.pad(d, r, constant_values=np.nan)
    st = np.stack([pad[i:i + H, j:j + W] for i in range(size) for j in range(size)], 0)
    return np.nanmedian(st, axis=0)

def normconv(d, sigma):
    m = np.isfinite(d).astype(np.float32); v = np.where(np.isfinite(d), d, 0).astype(np.float32)
    num = ndimage.gaussian_filter(v, sigma); den = ndimage.gaussian_filter(m, sigma)
    out = np.full_like(d, np.nan); okk = den > 1e-3; out[okk] = num[okk] / den[okk]
    return out

rep = open(os.path.join(a.out, 'snap_report.tsv'), 'a')
if rep.tell() == 0: rep.write('winding\tn_valid\tfrac_with_sheet\tfrac_snapped\tmedian_abs_off\tp90_abs_off\tfrac_rejected\tfrac_already_on\n')
start = 0
for wi, (w, M) in enumerate(meshes.items()):
    nv = len(M['vy']); prof = vals[start:start + nv * len(offs)].reshape(nv, len(offs)) > 0; start += nv * len(offs)
    # runs of True along axis 1 -> pick the run whose centre is nearest offset 0
    d = np.full(nv, np.nan, np.float32)
    padded = np.concatenate([np.zeros((nv, 1), bool), prof, np.zeros((nv, 1), bool)], 1).astype(np.int8)
    dif = np.diff(padded, axis=1)
    has = prof.any(1)
    for i in np.nonzero(has)[0]:
        st_ = np.nonzero(dif[i] == 1)[0]; en_ = np.nonzero(dif[i] == -1)[0]   # run = [st, en)
        centres = (st_ + en_ - 1) / 2.0 - K
        j = int(np.argmin(np.abs(centres)))
        # parabolic sub-voxel refinement on a [1,2,1]-smoothed profile around the run centre
        sm = np.convolve(prof[i].astype(np.float32), [1, 2, 1], mode='same')
        c = int(round(centres[j] + K)); c = min(max(c, 1), len(offs) - 2)
        y0_, y1_, y2_ = sm[c - 1], sm[c], sm[c + 1]; den = (y0_ - 2 * y1_ + y2_)
        frac = 0.5 * (y0_ - y2_) / den if abs(den) > 1e-6 else 0.0
        d[i] = centres[j] if abs(frac) > 0.5 else 0.5 * (centres[j] + (c - K + frac))
    D = np.full(M['xs'].shape, np.nan, np.float32); D[M['vy'], M['vx']] = d
    already_on = float((np.abs(d) <= 2.0).mean()) if nv else 0.0
    # regularise: reject jumps vs the 5x5 nan-median, fill, smooth
    med = nanmedian_filter(D, 5)
    rej = np.isfinite(D) & np.isfinite(med) & (np.abs(D - med) > a.jump)
    D2 = D.copy(); D2[rej] = np.nan
    filled = normconv(D2, max(a.sigma, 1.0))            # fill holes from neighbours
    D3 = np.where(np.isfinite(D2), D2, filled)
    D3 = normconv(D3, a.sigma)                           # final smoothing
    D3 = np.clip(D3, -K, K)
    apply = M['ok'] & np.isfinite(D3) & np.isfinite(filled)
    # a vertex whose whole neighbourhood had no sheet keeps its fitted position
    xs, ys, zs = M['xs'].copy(), M['ys'].copy(), M['zs'].copy()
    n = M['n']
    xs[apply] += D3[apply] * n[apply][:, 0]; ys[apply] += D3[apply] * n[apply][:, 1]; zs[apply] += D3[apply] * n[apply][:, 2]
    od = os.path.join(a.out, w); os.makedirs(od, exist_ok=True)
    for name, arr in (('x', xs), ('y', ys), ('z', zs)):
        arr = arr.astype(np.float32); arr[~M['valid']] = -1.0
        tifffile.imwrite(os.path.join(od, f'{name}.tif'), arr)
    meta = json.load(open(os.path.join(M['dir'], 'meta.json')))
    v = M['valid']
    meta['bbox'] = [[float(xs[v].min()), float(ys[v].min()), float(zs[v].min())], [float(xs[v].max()), float(ys[v].max()), float(zs[v].max())]]
    meta['source'] = meta.get('source', '') + ' + snap_meshes_local (offset along normal to nearest predicted sheet, regularised)'
    meta['snap'] = dict(max_off=K, jump=a.jump, sigma=a.sigma, frac_snapped=float(apply.sum() / max(1, v.sum())),
                        median_abs_off=float(np.nanmedian(np.abs(D3[apply]))) if apply.any() else None)
    json.dump(meta, open(os.path.join(od, 'meta.json'), 'w'), indent=4)
    line = (f'{w}\t{int(v.sum())}\t{has.mean():.3f}\t{apply.sum()/max(1,v.sum()):.3f}\t'
            f'{np.nanmedian(np.abs(D3[apply])) if apply.any() else float("nan"):.2f}\t{np.nanpercentile(np.abs(D3[apply]), 90) if apply.any() else float("nan"):.2f}\t'
            f'{rej.sum()/max(1,has.sum()):.3f}\t{already_on:.3f}')
    rep.write(line + '\n'); rep.flush(); print(line, flush=True)
print(f'done {len(meshes)} windings in {time.time()-t0:.0f}s -> {a.out}')
