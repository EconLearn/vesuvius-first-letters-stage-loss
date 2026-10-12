#!/usr/bin/env python
"""score_regions: pre-registered region test for the 343 calibration (and stage 2 on our surface).

Regions come from calib343_regions.py (letters_raw / nulls_raw, render-canvas pixel boxes [r0, r1, c0, c1]),
optionally shifted by --offset r,c when the ensemble canvas is a crop. For every ensemble map tag (d-06 ... d+06,
dmean, dmax) it reports, over VALID pixels: mean z inside the letter boxes, mean z inside each null set, p95 inside,
and the AUC of letter-box pixels vs null-box pixels.

PRE-REGISTERED (2026-10-10, before any 343 ink is computed by us):
  PASS_z     : mean ensemble z inside the letter boxes > mean inside EVERY null set, for the primary tag
               (default dmean = mean over the reduced depths d = -6..+6 step 2 of the 4-model z-mean,
               reading side, tile phases merged as his merge rule);
  SABOTAGE   : on the depth-shuffled render (same pipeline, d = -2, 0, +2), the letter advantage
               (letter - max null) must fall below 25 % of the unshuffled advantage at the same tag (dmean);
  OBSERVER   : the line-scale strips (display.py) must show the letters to an observer -- recorded by hand in
               observer.txt next to this score (this script cannot judge it and says so).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def masks(reg, H, W, off):
    def m(boxes):
        x = np.zeros((H, W), bool)
        for r0, r1, c0, c1 in boxes:
            r0, r1, c0, c1 = r0 - off[0], r1 - off[0], c0 - off[1], c1 - off[1]
            x[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = True
        return x
    return m(reg['letters_raw']), [m(b) for b in reg['nulls_raw']]


def auc(pos, neg):
    from scipy.stats import rankdata
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score_dir(ens, name, reg, off, tags):
    valid = np.load(Path(ens) / f'{name}__valid.npy')
    H, W = valid.shape
    lm, nms = masks(reg, H, W, off)
    out = {}
    for t in tags:
        f = Path(ens) / f'{name}__ensemble__{t}.npy'
        if not f.exists():
            continue
        E = np.load(f)
        li = E[lm & valid]
        nulls = [E[nm & valid] for nm in nms]
        allnull = np.concatenate(nulls) if nulls else np.array([])
        out[t] = dict(letter_mean=float(li.mean()) if li.size else float('nan'), letter_px=int(li.size),
                      letter_p95=float(np.percentile(li, 95)) if li.size else float('nan'),
                      null_means=[float(x.mean()) if x.size else float('nan') for x in nulls],
                      null_px=[int(x.size) for x in nulls],
                      auc_letter_vs_null=auc(li, allnull))
        o = out[t]
        o['advantage'] = o['letter_mean'] - max(o['null_means'])
        o['PASS_z'] = bool(all(o['letter_mean'] > x for x in o['null_means']))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ens', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--regions', required=True)
    ap.add_argument('--offset', default='0,0', help='canvas origin in region coordinates (r,c)')
    ap.add_argument('--primary', default='dmean')
    ap.add_argument('--sabotage-ens'); ap.add_argument('--sabotage-name')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    reg = json.loads(Path(a.regions).read_text())
    off = tuple(int(v) for v in a.offset.split(','))
    tags = [f'd{d:+03d}' for d in range(-15, 16)] + ['dmean', 'dmax']
    res = dict(primary=a.primary, regions=a.regions, scores=score_dir(a.ens, a.name, reg, off, tags))
    p = res['scores'].get(a.primary)
    res['PASS_z'] = bool(p and p['PASS_z'])
    if a.sabotage_ens:
        sab = score_dir(a.sabotage_ens, a.sabotage_name or a.name, reg, off, tags)
        res['sabotage'] = sab
        s = sab.get(a.primary)
        if p and s:
            res['sabotage_advantage_ratio'] = s['advantage'] / p['advantage'] if p['advantage'] > 0 else float('nan')
            res['SABOTAGE_OK'] = bool(p['advantage'] > 0 and s['advantage'] < 0.25 * p['advantage'])
    res['OBSERVER'] = 'REQUIRED: inspect the line-scale strips and record yes/no in observer.txt'
    res['PASS_automatic_parts'] = bool(res['PASS_z'] and res.get('SABOTAGE_OK', False))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('scores', 'sabotage')}, indent=1))
    if p:
        print('primary', a.primary, json.dumps(p))
    return 0


if __name__ == '__main__':
    sys.exit(main())
