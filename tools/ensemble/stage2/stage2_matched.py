#!/usr/bin/env python
"""Stage-2 scorer: the calibration's matched-null rule (score_matched.py, unchanged) applied on OUR q343 surface.

Input: the output dir of stage2_q343.py --run (regions_ours.json = his 18 letter boxes mapped onto our surface through
3D xyz, canvas offset; q343_letters.zarr; sweep/ (real, tag 'merged' = mean of tile phases 0,0 + 40,200) and
sweep_shuffled/ (depth-shuffled sabotage, tag 'ph0_0')).
Builds the same matched nulls as tools/matched_nulls.py, on the stage-2 canvas instead of his:
  letters   mapped letter boxes (canvas coords);  shift+/- half-letter-shifted boxes along the row;
  interline boxes of the letter's width and half its height, 16 px above and below, minus letters dilated 16 px,
            kept when >= 50 % remains;  ring: a 40 px band around each box minus letters dilated 8 px;
  context   valid pixels (CT > 0 in every rendered layer) with > 64 px of valid neighbourhood (chessboard), as ctx64.
Then score_matched.py with the calibration's specs (dnative, ink9_s43, ink9_s42, ens3; real d=-2..+2 and d=-4..+2,
sabotage d=-2..+2).

PRE-REGISTERED STAGE-2 PASS (written 2026-10-11, before any stage-2 ink exists):
  PASS iff score_matched's INK_EVIDENCED is true for BOTH dense_native ('dnative') AND the 3-model z-mean ('ens3') at
  d = -2..+2, each with >= 9 scored letters in both per-letter comparisons (interline I and ring R).
  INK_EVIDENCED (unchanged rule): for I and R, the per-letter (box - own null) mean has a 95 % bootstrap CI above 0,
  and the depth-shuffled sabotage does not (mean < 25 % of the real one, or CI includes 0).
  Fewer than 9 scored letters -> INCONCLUSIVE. Plus the observer check of the strips (recorded by hand).
usage: stage2_matched.py --stage2 DIR [--out DIR/matched] [--min-letters 9]"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ENS = Path(os.environ.get('ENS_LANE', str(Path(__file__).resolve().parents[1])))    # tools/ensemble
sys.path.insert(0, str(ENS))
from ens import common as C  # noqa: E402

GAP, RING, CTX = 16, 40, 64


def build(stage2, out):
    reg = json.loads((stage2 / 'regions_ours.json').read_text())
    (R0, _), (C0, _) = reg['canvas']['rows'], reg['canvas']['cols']
    arr, at = C.open_render(stage2 / 'q343_letters.zarr')
    n, H, W = arr.shape
    valid = C.valid_mask_all_layers(arr, k_range=at['ensemble_render'].get('rendered_k'))
    letters = []
    for r0, r1, c0, c1 in reg['letters_raw']:
        b = [max(0, r0 - R0), min(H, r1 - R0), max(0, c0 - C0), min(W, c1 - C0)]
        if b[1] > b[0] and b[3] > b[2]:
            letters.append([int(v) for v in b])
    lm = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in letters:
        lm[r0:r1, c0:c1] = True
    lm16 = ndimage.binary_dilation(lm, iterations=16); lm8 = ndimage.binary_dilation(lm, iterations=8)
    shp, shm, inter = [], [], []
    for r0, r1, c0, c1 in letters:
        w, h = c1 - c0, r1 - r0
        shp.append([r0, r1, min(W, c0 + w // 2), min(W, c1 + w // 2)])
        shm.append([r0, r1, max(0, c0 - w // 2), max(0, c1 - w // 2)])
        for rr0, rr1 in ((r1 + GAP, r1 + GAP + h // 2), (r0 - GAP - h // 2, r0 - GAP)):
            rr0, rr1 = max(0, rr0), min(H, rr1)
            if rr1 > rr0 and (~lm16[rr0:rr1, c0:c1]).mean() >= 0.5:
                inter.append([int(rr0), int(rr1), int(c0), int(c1)])
    ring = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in letters:
        ring[max(0, r0 - RING):min(H, r1 + RING), max(0, c0 - RING):min(W, c1 + RING)] = True
    ring &= ~lm8
    need = lm.copy()
    for bx in shp + shm + inter:
        need[bx[0]:bx[1], bx[2]:bx[3]] = True
    need |= ring
    dist = ndimage.distance_transform_cdt(np.pad(valid, 1), metric='chessboard')[1:-1, 1:-1]
    ctx = dist > CTX
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'ctx64.npy', ctx); np.save(out / 'need_score.npy', need & ctx & valid)
    res = dict(letters_raw=letters, shift_plus=shp, shift_minus=shm, interline=inter, render_shape=[int(n), H, W],
               canvas_offset=[R0, C0], gap=GAP, ring=RING, ctx=CTX, source='stage2_matched.py from ' + str(stage2),
               letters_px_in_context=int((lm & ctx & valid).sum()), need_score_px=int((need & ctx & valid).sum()))
    (out / 'regions_matched.json').write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage2', required=True); ap.add_argument('--out')
    ap.add_argument('--min-letters', type=int, default=9)
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--models', default='ink9_s42,ink9_s43,dnative', help='members swept by stage2 (ens3 = their z-mean)')
    a = ap.parse_args()
    s2 = Path(a.stage2); out = Path(a.out) if a.out else s2 / 'matched'
    reg = build(s2, out)
    print(json.dumps({k: v for k, v in reg.items() if k not in ('letters_raw', 'shift_plus', 'shift_minus', 'interline')}))
    args = []
    members = a.models.split(',')
    specs = [(m, m) for m in members] + [('ens3', a.models)]
    have_d4 = all((s2 / 'sweep' / 'maps' / m / f'{m}__reverse__merged__d-04.tif').exists() for m in members)
    if not have_d4:
        print('note: no d=-4 maps (stage 2 swept fewer depths); the secondary d=-4..+2 arm is skipped')
    for lab, mods in specs:
        if have_d4:
            args += ['--real', f'{lab}@d-4..2={s2}/sweep:{mods}:merged:-4..2:2']
        args += ['--real', f'{lab}={s2}/sweep:{mods}:merged:-2..2:2',
                 '--sab', f'{lab}={s2}/sweep_shuffled:{mods}:ph0_0:-2..2:2']
    cmd = [a.py, str(ENS / 'score_matched.py'), '--render', str(s2 / 'q343_letters.zarr'), '--regions',
           str(out / 'regions_matched.json'), '--ctx', str(out / 'ctx64.npy'), '--need', str(out / 'need_score.npy'),
           '--out', str(out / 'score_matched.json')] + args
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f'score_matched failed rc={r.returncode}')
    sc = json.loads((out / 'score_matched.json').read_text())
    arms = {}
    for lab in ('dnative', 'ens3'):
        v = sc['verdict'].get(lab, {})
        nI = sc['real'][lab]['per_letter_I']['n']; nR = sc['real'][lab]['per_letter_R']['n']
        arms[lab] = dict(INK_EVIDENCED=bool(v.get('INK_EVIDENCED')), n_letters_I=nI, n_letters_R=nR,
                         enough_letters=bool(nI >= a.min_letters and nR >= a.min_letters),
                         real_I=sc['real'][lab]['per_letter_I'], real_R=sc['real'][lab]['per_letter_R'],
                         sab_I=sc['sab'][lab]['per_letter_I'], sab_R=sc['sab'][lab]['per_letter_R'])
    if not all(x['enough_letters'] for x in arms.values()):
        verdict = 'INCONCLUSIVE'
    else:
        verdict = 'PASS' if all(x['INK_EVIDENCED'] for x in arms.values()) else 'FAIL'
    res = dict(verdict=verdict, rule=__doc__.split('PRE-REGISTERED STAGE-2 PASS')[1].split('usage:')[0].strip(),
               arms=arms, other_members={k: v for k, v in sc['verdict'].items() if k not in arms})
    (out / 'PASS.json').write_text(json.dumps(res, indent=1))
    print(f'STAGE2 VERDICT: {verdict}  ' + '  '.join(
        f"{k}: ink={x['INK_EVIDENCED']} I={x['real_I']['mean']} {x['real_I']['ci95']} R={x['real_R']['mean']} "
        f"{x['real_R']['ci95']} n={x['n_letters_I']}/{x['n_letters_R']}" for k, x in arms.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
