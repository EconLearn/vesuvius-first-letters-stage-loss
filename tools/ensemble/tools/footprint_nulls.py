#!/usr/bin/env python
"""Null boxes that need NO CT beyond the letters' own chunks (CPU calibration under a 1.0 GB CT cap).

Geometry only, decided before any 343 ink is computed. Rule:
  * the letters' render mask = letter boxes + MARGIN px; its CT chunk set S_L is computed exactly as render66 reads
    (32 px render tiles, layers k0..k1-1, superset over all valid pixels of each touched tile);
  * a render tile is 'free' when its own chunk set is inside S_L;
  * null boxes are NULL_SIZE squares (default 64 px) whose box + MARGIN lies entirely in free, valid pixels farther
    than 40 px from any letter; centres scanned in raster order on an 8 px grid, accepted when disjoint from the boxes
    already accepted; assigned round-robin to the null sets.
  (Revised 2026-10-11 under a 1.0 GB CT cap, before any 343 ink was computed: letters + any
   letter-sized null set needed 1.6-2.1 GB. Same-size partner nulls are kept in regions.json for the GPU run.)
usage: footprint_nulls.py REL REGIONS.json OUT.json [--margin 64] [--k-range 18:49] [--tile 32] [--n-null 3]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ens import common as C  # noqa: E402

NAME = 'concat_w047-w048_R5B2_z9500-11000'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rel'); ap.add_argument('regions'); ap.add_argument('out')
    ap.add_argument('--margin', type=int, default=64); ap.add_argument('--k-range', default='18:49')
    ap.add_argument('--tile', type=int, default=32); ap.add_argument('--n-null', type=int, default=3)
    ap.add_argument('--null-size', type=int, default=64)
    a = ap.parse_args()
    reg = json.loads(Path(a.regions).read_text())
    n, H, W = reg['render_shape']
    k0, k1 = (int(v) for v in a.k_range.split(':'))
    L = C.use_volume('PHerc0343')
    P, valid_nodes, meta = L.load_tifxyz(C.rel_file(a.rel, f'{NAME}.tifxyz'))
    N = L.vertex_normals(P); gs = float(meta['scale'][0])
    pts, nrm = L.surface_at_pixels(P, N, np.arange(H), np.arange(W), gs)
    ok = np.isfinite(pts).all(-1) & np.isfinite(nrm).all(-1)
    offs = (np.arange(k0, k1) - 0.5 * (n - 1))
    T = a.tile
    nty, ntx = -(-H // T), -(-W // T)
    keys = {}
    for ty in range(nty):
        for tx in range(ntx):
            sl = (slice(ty * T, (ty + 1) * T), slice(tx * T, (tx + 1) * T))
            v = ok[sl]
            if not v.any():
                continue
            p = pts[sl][v]; d = nrm[sl][v]
            lo = np.minimum((p + offs[0] * d).min(0), (p + offs[-1] * d).min(0))
            hi = np.maximum((p + offs[0] * d).max(0), (p + offs[-1] * d).max(0))
            lo = np.floor(lo).astype(int) - 1; hi = np.ceil(hi).astype(int) + 2
            ks = set()
            for cz in range(max(0, lo[2]) // 128, (hi[2] - 1) // 128 + 1):
                for cy in range(max(0, lo[1]) // 128, (hi[1] - 1) // 128 + 1):
                    for cx in range(max(0, lo[0]) // 128, (hi[0] - 1) // 128 + 1):
                        ks.add((cz, cy, cx))
            keys[(ty, tx)] = frozenset(ks)
    letters = reg['letters_raw']
    lmask = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in letters:
        lmask[r0:r1, c0:c1] = True
    lrend = ndimage.maximum_filter(lmask, size=2 * a.margin + 1)
    tl = {(ty, tx) for (ty, tx) in keys if lrend[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].any()}
    SL = set().union(*[keys[t] for t in tl])
    free = np.zeros((nty, ntx), bool)
    for t, ks in keys.items():
        free[t] = ks <= SL
    # pixel-level 'free' map (a pixel is free if its tile is free); invalid tiles count as not free
    freepx = np.repeat(np.repeat(free, T, 0), T, 1)[:H, :W]
    vmask = ok
    forbid = ndimage.binary_dilation(lmask, iterations=40)
    cand = freepx & vmask & ~forbid
    # chessboard distance to the nearest non-candidate pixel: a box of size S centred here, plus MARGIN, is inside cand
    dist = ndimage.distance_transform_cdt(np.pad(cand, 1), metric='chessboard')[1:-1, 1:-1]
    S = a.null_size
    mg = a.margin
    need = S // 2 + mg
    ys, xs = np.nonzero(dist[::8, ::8] >= need + 1)
    order = np.lexsort((xs, ys))
    taken = np.zeros((H, W), bool)
    boxes = []
    for i in order:
        cy, cx = int(ys[i]) * 8, int(xs[i]) * 8
        r0, c0 = cy - S // 2, cx - S // 2
        if taken[r0:r0 + S, c0:c0 + S].any():
            continue
        boxes.append([r0, r0 + S, c0, c0 + S]); taken[r0:r0 + S, c0:c0 + S] = True
    nulls = [boxes[k::a.n_null] for k in range(a.n_null)]
    short = []
    out = dict(reg, nulls_raw=nulls, nulls_rule=__doc__.split('Geometry only')[1].strip(),
               nulls_preregistered_full_canvas=reg['nulls_raw'], letters_ct_chunks=len(SL),
               letters_ct_gb=round(len(SL) * 128 ** 3 / 1e9, 3), footprint_margin=mg, k_range=[k0, k1],
               letters_short_of_partners=short, free_valid_px=int(cand.sum()), null_size=S)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(letters_ct_chunks=len(SL), letters_ct_gb=out['letters_ct_gb'],
                          nulls_per_set=[len(x) for x in nulls], short=short,
                          free_valid_px=out['free_valid_px'])))


if __name__ == '__main__':
    main()
