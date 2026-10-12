#!/usr/bin/env python
"""Compare model maps of two runs pixel by pixel on a mask (default: pixels where either map is non-zero or the need
mask is set). Criterion: |diff| <= 1 grey level on >= 99.99 % of pixels, or bit-identical.
usage: compare_maps.py --need NEED.npy A.tif:B.tif [A2.tif:B2.tif ...] [--json OUT]"""
import argparse
import json

import numpy as np
import tifffile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+')
    ap.add_argument('--need')
    ap.add_argument('--json')
    a = ap.parse_args()
    need = np.load(a.need).astype(bool) if a.need else None
    out = []
    tot = dict(px=0, identical=0, within1=0, maxdiff=0)
    for pr in a.pairs:
        fa, fb = pr.split(':')
        A, B = tifffile.imread(fa), tifffile.imread(fb)
        if A.shape != B.shape:
            raise SystemExit(f'shape mismatch {A.shape} {B.shape}')
        m = need if need is not None else np.ones(A.shape, bool)
        m = m[:A.shape[0], :A.shape[1]] if m.shape != A.shape else m
        d = np.abs(A.astype(np.int64) - B.astype(np.int64))[m]
        whole = int((A != B).sum())
        r = dict(a=fa, b=fb, px=int(m.sum()), identical=int((d == 0).sum()), within1=int((d <= 1).sum()),
                 maxdiff=int(d.max()) if d.size else 0, mean_abs=float(d.mean()) if d.size else 0.0,
                 differing_px_whole_map=whole)
        r['frac_identical'] = r['identical'] / max(1, r['px']); r['frac_within1'] = r['within1'] / max(1, r['px'])
        out.append(r)
        for k in ('px', 'identical', 'within1'):
            tot[k] += r[k]
        tot['maxdiff'] = max(tot['maxdiff'], r['maxdiff'])
        print(f"{fa.split('/')[-1]}: px {r['px']} identical {r['frac_identical']:.6f} within1 {r['frac_within1']:.6f} "
              f"max {r['maxdiff']} (whole-map differing px {whole})")
    tot['frac_identical'] = tot['identical'] / max(1, tot['px']); tot['frac_within1'] = tot['within1'] / max(1, tot['px'])
    tot['PASS'] = bool(tot['frac_within1'] >= 0.9999)
    tot['BIT_IDENTICAL'] = bool(tot['identical'] == tot['px'])
    print('TOTAL', json.dumps(tot))
    if a.json:
        json.dump(dict(pairs=out, total=tot), open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
