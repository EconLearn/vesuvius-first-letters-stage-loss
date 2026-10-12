#!/usr/bin/env python
"""Ranking control on PHerc0343 (pre-registered 2026-10-11 ~23:05 UTC, before computing).

Does the band-screen tile ranking (512-px tiles with >= 50 % valid px; score = p95 - p50 of the mean over d = -2, 0, +2
of dense_native unit maps, reverse, phase 0,0) put the winner's letters in the top 10 %?
Data: the stage-2 output dir of stage2_q343.py --run (our surface window, maps from its sweep).
Letter tile = a tile that contains the centre of one of his 18 boxes (mapped onto our surface).
PASS iff >= 50 % of letter tiles are in the top 10 % of valid tiles AND one-sided binomial p < 0.01 (chance 10 %).
Also reported: AUC (letter tiles vs other tiles), letter-tile rank percentiles, and the same on the depth-shuffled
sabotage maps (expected: no enrichment). Caveat: other tiles in this window include unlabelled text, which biases
toward FAIL.
--edge E scores only pixels >= E px (Chebyshev) inside the valid mask, and skips tiles with < 25 % such pixels.

usage: rankctl_343.py [--stage2 DIR] [--edge 0] [--out FILE]"""
import argparse
import glob
import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ens import adapters as A  # noqa: E402
from ens import common as C  # noqa: E402

T = 512


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage2', default=str(C.WORK / 'stage2x'), help='output dir of stage2_q343.py --run')
    ap.add_argument('--edge', type=int, default=0)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    W = Path(a.stage2)
    reg = json.load(open(W / 'matched' / 'regions_matched.json'))['letters_raw']
    z = zarr.open(str(W / 'q343_letters.zarr'), mode='r')
    arr = z if hasattr(z, 'shape') else z[list(z.array_keys())[0]]
    valid = np.asarray(arr[33]) > 0
    E = a.edge
    interior = (ndimage.minimum_filter(valid.astype(np.uint8), size=2 * E + 1, mode='constant', cval=0) > 0
                if E else valid)

    def score_map(pat):
        acc = None
        for d in (-2, 0, 2):
            f = glob.glob(pat.format(d=f'{d:+03d}'))
            assert len(f) == 1, (pat, d, f)
            u = A.VillaAdapter.to_unit(tifffile.imread(f[0]))
            acc = u if acc is None else acc + u
        return acc / 3

    def rank(u):
        H, Wd = u.shape
        rows = []
        for ty in range(0, H, T):
            for tx in range(0, Wd, T):
                vv = valid[ty:ty + T, tx:tx + T]
                if vv.size == 0 or vv.mean() < 0.5:
                    continue
                ee = interior[ty:ty + T, tx:tx + T]
                if ee.mean() < 0.25:
                    continue
                x = u[ty:ty + T, tx:tx + T][ee]
                p50, p95 = np.percentile(x, [50, 95])
                rows.append((ty // T, tx // T, float(p95 - p50)))
        return rows

    cent = {((r0 + r1) // 2 // T, (c0 + c1) // 2 // T) for r0, r1, c0, c1 in reg}

    def report(name, rows):
        s = np.array([r[2] for r in rows])
        lt = np.array([(r[0], r[1]) in cent for r in rows])
        n = len(rows)
        nsel = max(1, math.ceil(0.10 * n))
        thr = np.sort(s)[::-1][nsel - 1]
        sel = s >= thr
        k = int((lt & sel).sum())
        m = int(lt.sum())
        p = sum(comb(m, j) * 0.1 ** j * 0.9 ** (m - j) for j in range(k, m + 1))
        pos, neg = s[lt], s[~lt]
        auc = (float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
               if len(pos) and len(neg) else None)
        pct = sorted(round(float((s < v).mean()) * 100, 1) for v in pos)
        out = dict(arm=name, edge=E, n_valid_tiles=n, n_selected=nsel, letter_tiles=m, letter_tiles_in_top10=k,
                   frac=round(k / max(m, 1), 3), binom_p=float(p), auc=auc, letter_pct_ranks=pct,
                   PASS=bool(m and k / m >= 0.5 and p < 0.01))
        print(json.dumps(out))
        return out

    res = [report('real', rank(score_map(str(W / 'sweep' / 'maps' / 'dnative' / 'dnative__reverse__ph0_0__d{d}.tif'))))]
    sab = [f for f in glob.glob(str(W / '**' / 'dnative__reverse__ph0_0__d+00.tif'), recursive=True)
           if '/sweep/maps/' not in f.replace('\\', '/')]
    if sab:
        res.append(report('sabotage', rank(score_map(sab[0].replace('d+00.tif', 'd{d}.tif')))))
    out = Path(a.out) if a.out else W / f'rankctl_343_e{E}.json'
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
