#!/usr/bin/env python
"""render66: N-layer (default 66) surface-volume render of a tifxyz region with our validated vc_render_tifxyz port
(tools/g12lib.py), in Nieuwlaar's depth convention: layer k at offset k - 32.5 along the unit normal,
d = k - 33, +d pointing AWAY from the scroll axis.

Normal sign: --normal-sign auto (default) measures, against the scroll's umbilicus, the fraction of pixels whose
normal points outward and flips the normals when that fraction is < 0.5; it refuses (exit 3) when the fraction is
between 0.2 and 0.8 (a mesh whose normal sign is not consistent, e.g. a fold), unless --force. keep / flip override.

Region: --rows r0:r1 --cols c0:c1 in render pixels of the full tifxyz render at --render-scale (grid coordinate =
(pixel + 0.5) * grid_scale / render_scale, measured convention). Or --geo NPZ (pts, nrm arrays, e.g. G1G2 geo.npz)
with optional --rows/--cols inside it.

Verification: --verify28 renders the same pixels with 28 layers through the old code path and requires the
overlapping layers (66-layer k = 28-layer j + 19) to be bit-identical (exit 4 otherwise).

Outputs: <out>.zarr (group, array '0' (N, H, W) uint8, attrs = convention + provenance) and, with --save-pts,
<out>.pts.npy (H, W, 3) float32 XYZ surface points (for 3D mapping between surfaces).
CT chunks are read from the local mirror only; fetching needs ENS_ALLOW_FETCH=1 (see ens/common.use_volume).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens import common as C  # noqa: E402


def rng(s):
    a, b = s.split(':')
    return int(a), int(b)


def plan_chunks(L, pts, nrm, a, flip):
    """Exact emulation of g12lib.render_pixels/_render_tile read boxes -> chunk keys; counts the ones not mirrored.
    Writes <out>.chunks.txt (missing keys) so an approved CT fetch can be exactly this list."""
    k0, k1 = a.krange
    offs = (np.arange(k0, k1) - 0.5 * (a.n - 1)) * a.step
    d = -nrm if flip else nrm
    valid = np.isfinite(pts).all(-1) & np.isfinite(d).all(-1)
    keys = set()
    H, W = valid.shape
    S0 = L.SHAPE
    for i0 in range(0, H, a.tile):
        for j0 in range(0, W, a.tile):
            v = valid[i0:i0 + a.tile, j0:j0 + a.tile]
            if not v.any():
                continue
            p = pts[i0:i0 + a.tile, j0:j0 + a.tile][v]; n = d[i0:i0 + a.tile, j0:j0 + a.tile][v]
            lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
            for o in (offs[0], offs[-1]):                       # extremes of a straight segment bound it
                q = p + o * n
                lo = np.minimum(lo, q.min(0)); hi = np.maximum(hi, q.max(0))
            lo = np.floor(lo).astype(int) - 1; hi = np.ceil(hi).astype(int) + 2
            if np.any(hi - lo > 1500):
                continue
            z0, z1, y0, y1, x0, x1 = lo[2], hi[2], lo[1], hi[1], lo[0], hi[0]
            for cz in range(max(0, z0) // 128, (min(S0[0], z1) - 1) // 128 + 1):
                for cy in range(max(0, y0) // 128, (min(S0[1], y1) - 1) // 128 + 1):
                    for cx in range(max(0, x0) // 128, (min(S0[2], x1) - 1) // 128 + 1):
                        keys.add((cz, cy, cx))
    absent_f = L.MIRROR.parent / '.absent_L0.txt'
    absent = set(absent_f.read_text().split()) if absent_f.exists() else set()
    miss = sorted(k for k in keys if not (L.MIRROR / f'{k[0]}/{k[1]}/{k[2]}').exists()
                  and f'{k[0]}/{k[1]}/{k[2]}' not in absent)
    lst = Path(str(a.out).replace('.zarr', '') + '.chunks.txt')
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text('\n'.join(f'{k[0]}/{k[1]}/{k[2]}' for k in miss) + '\n')
    print(json.dumps(dict(tile=a.tile, chunks_read=len(keys), chunks_missing=len(miss),
                          missing_gb=round(len(miss) * 128 ** 3 / 1e9, 3), list=str(lst))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scroll', required=True, choices=sorted(C.VOLUMES))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--tifxyz')
    src.add_argument('--geo', help='npz with pts (H,W,3) and nrm (H,W,3), XYZ, as written by g12_g2 stage_org')
    ap.add_argument('--rows'); ap.add_argument('--cols')
    ap.add_argument('--render-scale', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=C.N66)
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--normal-sign', choices=('auto', 'keep', 'flip'), default='auto')
    ap.add_argument('--umbilicus', help='override the scroll default umbilicus.json')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--tile', type=int, default=64, help='render tile (px); smaller = tighter CT read boxes, same values')
    ap.add_argument('--out', required=True, help='output .zarr path')
    ap.add_argument('--save-pts', action='store_true')
    ap.add_argument('--verify28', action='store_true')
    ap.add_argument('--k-range', help='render only layers k0:k1 (others stay 0; recorded in attrs rendered_k)')
    ap.add_argument('--pixel-mask', help='bool (H, W) npy on the requested rows/cols: render only these pixels')
    ap.add_argument('--plan-chunks', action='store_true',
                    help='only list the CT chunks this render would read and how many are not mirrored (no render)')
    a = ap.parse_args(argv)
    a.krange = rng(a.k_range) if a.k_range else (0, a.n)
    t0 = time.time()
    L = C.use_volume(a.scroll)
    if a.tifxyz:
        P, valid, meta = L.load_tifxyz(a.tifxyz)
        N = L.vertex_normals(P)
        gs = float(meta['scale'][0])
        H = int(np.floor(P.shape[0] / gs * a.render_scale)); W = int(np.floor(P.shape[1] / gs * a.render_scale))
        r0, r1 = rng(a.rows) if a.rows else (0, H)
        c0, c1 = rng(a.cols) if a.cols else (0, W)
        pts, nrm = L.surface_at_pixels(P, N, np.arange(r0, r1), np.arange(c0, c1), gs, a.render_scale)
        prov = dict(tifxyz=str(Path(a.tifxyz).resolve()), grid_scale=gs, render_scale=a.render_scale,
                    full_render_hw=[H, W], rows=[r0, r1], cols=[c0, c1])
    else:
        g = np.load(a.geo)
        pts, nrm = g['pts'], g['nrm']
        r0, r1 = rng(a.rows) if a.rows else (0, pts.shape[0])
        c0, c1 = rng(a.cols) if a.cols else (0, pts.shape[1])
        pts, nrm = pts[r0:r1, c0:c1].astype(np.float32), nrm[r0:r1, c0:c1].astype(np.float32)
        nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
        prov = dict(geo=str(Path(a.geo).resolve()), rows=[r0, r1], cols=[c0, c1])
    if a.pixel_mask:
        pm = np.load(a.pixel_mask).astype(bool)
        if pm.shape != pts.shape[:2]:
            raise SystemExit(f'pixel mask {pm.shape} != region {pts.shape[:2]}')
        pts = pts.copy(); pts[~pm] = np.nan
        prov['pixel_mask'] = str(Path(a.pixel_mask).resolve())
    umb = C.load_umbilicus(a.umbilicus or C.VOLUMES[a.scroll]['umbilicus'])
    fout, npx = C.outward_fraction(pts, nrm, umb)
    if a.normal_sign == 'auto':
        if 0.2 < fout < 0.8 and not a.force:
            print(f'normal sign ambiguous: outward fraction {fout:.3f} of {npx} px; use --normal-sign or --force')
            return 3
        flip = fout < 0.5
    else:
        flip = a.normal_sign == 'flip'
    fout_after = 1 - fout if flip else fout
    print(f'region {pts.shape[:2]} valid {npx} px; outward fraction (mesh normals) {fout:.4f} -> flip={flip}; '
          f'+d outward on {fout_after:.4f} of pixels', flush=True)
    if a.plan_chunks:
        return plan_chunks(L, pts, nrm, a, flip)
    k0, k1 = a.krange
    m = k1 - k0
    eo = (k0 + 0.5 * (m - 1) - 0.5 * (a.n - 1)) * a.step      # local offsets (i-(m-1)/2)*step + eo == (k-(n-1)/2)*step
    sub, v = L.render_pixels(L.ChunkMirror(), pts, nrm, m, a.step, flip, tile=a.tile, extra_offset=eo, procs=a.procs)
    if m == a.n:
        vol = sub
    else:
        vol = np.zeros((a.n,) + sub.shape[1:], np.uint8); vol[k0:k1] = sub
        del sub
    t_render = time.time() - t0
    attrs = dict(ensemble_render=dict(
        n_layers=a.n, step=a.step, layer_offset='(k - (n-1)/2) * step along unit normal',
        depth_convention='d = k - n//2 (d = k - 33 for n = 66); +d away from the scroll axis',
        d0_layer=a.n // 2, flip_normals=bool(flip), outward_fraction_mesh=fout, outward_fraction_plusd=fout_after,
        scroll=a.scroll, valid_px=int(v.sum()), secs=round(t_render, 1), renderer='tools/g12lib.py',
        rendered_k=[int(a.krange[0]), int(a.krange[1])], tile=a.tile,
        **prov))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    if a.verify28:
        if a.n != C.N66 or a.step != 1.0:
            print('--verify28 needs n=66, step=1'); return 2
        v28, _ = L.render_pixels(L.ChunkMirror(), pts, nrm, 28, 1.0, flip, procs=a.procs)   # default tile 192
        d = np.abs(vol[19:47].astype(np.int16) - v28.astype(np.int16))
        res = dict(identical=bool((d == 0).all()), max_abs_diff=int(d.max()), frac_identical=float((d == 0).mean()),
                   compared='66-layer k=19..46 vs 28-layer j=0..27')
        attrs['ensemble_render']['verify28'] = res
        print('verify28', json.dumps(res), flush=True)
        if not res['identical']:
            rc = 4
    C.write_render(out, vol, attrs)
    if a.save_pts:
        np.save(str(out).replace('.zarr', '') + '.pts.npy', pts.astype(np.float32))
    print(f'wrote {out} {vol.shape} in {time.time() - t0:.0f}s (render {t_render:.0f}s)', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
