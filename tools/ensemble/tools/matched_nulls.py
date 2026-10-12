#!/usr/bin/env python
"""Matched nulls on the same text rows (2026-10-11), within the CT already fetched.

1. render mask = every valid pixel of every 32 px render tile whose CT chunk set (layers k0..k1-1, superset over
   the tile) is already mirrored -> the largest region renderable with no new CT;
2. regions (raw render px, geometry only):
   letters   his 18 letter boxes;
   shift+/- each letter box shifted along its row by +/- half its width (same row, same papyrus, misaligned);
   interline for each letter box, a box of the same width and half its height placed GAP px above and below it
             (reading orientation), minus all letter boxes dilated by 16 px; kept when >= 50 % remains;
   ring      a RING px band around each letter box minus all letter boxes dilated by 8 px (within-box contrast);
3. full-context pixels = need pixels whose (2*CTX+1)^2 neighbourhood is entirely rendered (CTX = 127 = one villa
   patch): only these are scored, for every region type alike.
Outputs in OUT: render2.npy, need2.npy, context2.npy, regions_matched.json (boxes + per-letter ring/interline masks
are rebuilt from the json by score_matched.py).
usage: matched_nulls.py REL REGIONS.json FETCHED_LIST OUT [--k-range 18:49] [--ring 40] [--gap 16]"""
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
    ap.add_argument('rel'); ap.add_argument('regions'); ap.add_argument('fetched'); ap.add_argument('out')
    ap.add_argument('--k-range', default='18:49'); ap.add_argument('--tile', type=int, default=32)
    ap.add_argument('--ring', type=int, default=40); ap.add_argument('--gap', type=int, default=16)
    ap.add_argument('--ctx', type=int, default=127)
    a = ap.parse_args()
    reg = json.loads(Path(a.regions).read_text())
    n, H, W = reg['render_shape']
    k0, k1 = (int(v) for v in a.k_range.split(':'))
    L = C.use_volume('PHerc0343')
    have = set()
    for l in Path(a.fetched).read_text().split('\n'):
        if l.strip():
            have.add(tuple(int(v) for v in l.split()[0].split('/')))
    for p in L.MIRROR.glob('*/*/*'):                         # anything else already mirrored
        try:
            have.add(tuple(int(v) for v in p.relative_to(L.MIRROR).parts))
        except ValueError:
            pass
    P, vn, meta = L.load_tifxyz(C.rel_file(a.rel, f'{NAME}.tifxyz'))
    N = L.vertex_normals(P); gs = float(meta['scale'][0])
    pts, nrm = L.surface_at_pixels(P, N, np.arange(H), np.arange(W), gs)
    ok = np.isfinite(pts).all(-1) & np.isfinite(nrm).all(-1)
    offs = np.arange(k0, k1) - 0.5 * (n - 1)
    T = a.tile
    free = np.zeros((-(-H // T), -(-W // T)), bool)
    for ty in range(free.shape[0]):
        for tx in range(free.shape[1]):
            sl = (slice(ty * T, (ty + 1) * T), slice(tx * T, (tx + 1) * T))
            v = ok[sl]
            if not v.any():
                continue
            p = pts[sl][v]; d = nrm[sl][v]
            lo = np.floor(np.minimum((p + offs[0] * d).min(0), (p + offs[-1] * d).min(0))).astype(int) - 1
            hi = np.ceil(np.maximum((p + offs[0] * d).max(0), (p + offs[-1] * d).max(0))).astype(int) + 2
            ks = {(cz, cy, cx) for cz in range(lo[2] // 128, (hi[2] - 1) // 128 + 1)
                  for cy in range(lo[1] // 128, (hi[1] - 1) // 128 + 1) for cx in range(lo[0] // 128, (hi[0] - 1) // 128 + 1)}
            free[ty, tx] = ks <= have
    render = np.repeat(np.repeat(free, T, 0), T, 1)[:H, :W] & ok
    letters = reg['letters_raw']
    lm = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in letters:
        lm[r0:r1, c0:c1] = True
    lm8 = ndimage.binary_dilation(lm, iterations=8); lm16 = ndimage.binary_dilation(lm, iterations=16)
    shp, shm, inter = [], [], []
    for r0, r1, c0, c1 in letters:
        w, h = c1 - c0, r1 - r0
        shp.append([r0, r1, min(W, c0 + w // 2), min(W, c1 + w // 2)])
        shm.append([r0, r1, max(0, c0 - w // 2), max(0, c1 - w // 2)])
        for rr0, rr1 in ((r1 + a.gap, r1 + a.gap + h // 2), (r0 - a.gap - h // 2, r0 - a.gap)):
            rr0, rr1 = max(0, rr0), min(H, rr1)
            if rr1 <= rr0:
                continue
            m = ~lm16[rr0:rr1, c0:c1]
            if m.mean() >= 0.5:
                inter.append([int(rr0), int(rr1), int(c0), int(c1)])
    ring = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in letters:
        R = a.ring
        ring[max(0, r0 - R):min(H, r1 + R), max(0, c0 - R):min(W, c1 + R)] = True
    ring &= ~lm8
    need = lm.copy()
    for bx in shp + shm + inter:
        need[bx[0]:bx[1], bx[2]:bx[3]] = True
    need |= ring
    need &= ok
    # full context: the whole (2*ctx+1)^2 square around the pixel is rendered (chessboard distance to unrendered)
    dist = ndimage.distance_transform_cdt(np.pad(render, 1), metric='chessboard')[1:-1, 1:-1]
    ctx = dist > a.ctx
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'render2.npy', render); np.save(out / 'need2.npy', need); np.save(out / 'context2.npy', ctx)
    np.save(out / 'ring2.npy', ring & ok)

    def frac(m):
        return round(float((m & ctx).sum() / max(1, (m & ok).sum())), 3)
    im = np.zeros((H, W), bool)
    for bx in inter:
        im[bx[0]:bx[1], bx[2]:bx[3]] = True
    im &= ~lm16
    spm = np.zeros((H, W), bool)
    for bx in shp + shm:
        spm[bx[0]:bx[1], bx[2]:bx[3]] = True
    np.save(out / 'interline2.npy', im & ok)
    res = dict(reg, shift_plus=shp, shift_minus=shm, interline=inter, ring_px=int((ring & ok).sum()),
               render_px=int(render.sum()), need_px=int(need.sum()), k_range=[k0, k1], ring=a.ring, gap=a.gap,
               ctx=a.ctx, full_context_fraction=dict(letters=frac(lm), shift=frac(spm), interline=frac(im),
                                                       ring=frac(ring)),
               rule=__doc__.split('Outputs in OUT')[0])
    (out / 'regions_matched.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(render_px=res['render_px'], need_px=res['need_px'], interline_boxes=len(inter),
                          full_context_fraction=res['full_context_fraction'])))


if __name__ == '__main__':
    main()
