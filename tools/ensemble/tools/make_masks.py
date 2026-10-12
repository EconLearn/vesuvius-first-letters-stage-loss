#!/usr/bin/env python
"""Pixel masks on his full render canvas for the restricted (CPU) calibration, from regions.json:
  need_score.npy   letter boxes dilated by --letter-dilate px (closes the gaps along a row for display) + the
                   selected null sets' boxes: the only pixels whose ink values are computed exactly / used;
  need_render.npy  need_score dilated by --margin px (>= 128 = one villa patch, so every patch that touches a
                   need pixel is fully rendered -> need pixels are identical to a whole-canvas run);
usage: make_masks.py REGIONS.json OUTDIR [--sets letters,0,1,2] [--margin 128] [--letter-dilate 24]"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('regions'); ap.add_argument('out')
ap.add_argument('--sets', default='letters,0,1,2')
ap.add_argument('--margin', type=int, default=128)
ap.add_argument('--letter-dilate', type=int, default=24)
a = ap.parse_args()
reg = json.loads(Path(a.regions).read_text())
n, H, W = reg['render_shape']
sets = a.sets.split(',')


def boxes_mask(boxes, grow=0):
    m = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in boxes:
        m[max(0, r0 - grow):min(H, r1 + grow), max(0, c0 - grow):min(W, c1 + grow)] = True
    return m


need = np.zeros((H, W), bool)
if 'letters' in sets:
    need |= boxes_mask(reg['letters_raw'], a.letter_dilate)
for s in sets:
    if s.isdigit():
        need |= boxes_mask(reg['nulls_raw'][int(s)])
# square dilation by margin (separable max filter)
rend = ndimage.maximum_filter(need, size=2 * a.margin + 1)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.save(out / 'need_score.npy', need)
np.save(out / 'need_render.npy', rend)
info = dict(sets=sets, margin=a.margin, letter_dilate=a.letter_dilate, need_px=int(need.sum()),
            render_px=int(rend.sum()), canvas=[H, W])
(out / 'masks.json').write_text(json.dumps(info, indent=1))
print(json.dumps(info))
