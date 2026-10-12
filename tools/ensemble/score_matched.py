#!/usr/bin/env python
"""Matched-null scoring (2026-10-11), real vs depth-shuffled sabotage.

Regions (from tools/matched_nulls.py; every pixel needs >= 64 px of rendered context and valid CT):
  L  his letter boxes
  I  interline: boxes of the letter's width, half its height, 16 px above and below it, minus letters dilated 16 px
  R  ring: a 40 px band around each letter box minus letters dilated 8 px (within-box contrast)
  S  half-letter-shifted boxes along the row (contain neighbouring letters' ink: secondary)
  Sg the same shifted boxes minus letters dilated 8 px (the inter-letter gaps / row ends)
Per member map (z over the scored pixels, his rule restricted to them): mean z per region, L - null, AUC, and
per-letter paired differences (letter box vs ITS OWN interline boxes / ring) with a bootstrap CI over letters.

PRE-REGISTERED here, before these scores were computed:
  INK EVIDENCED for a member iff, for BOTH matched nulls I and R, the real per-letter difference has a 95 % CI
  above 0 AND the sabotage (shuffled layers, same pixels, same depths) does not: its mean difference is
  < 25 % of the real one or its CI includes 0.
usage: score_matched.py --render R --regions regions_matched.json --ctx ctx64.npy --need need_score2.npy
                        --real 'label=MAPDIR:models:tag:depths' --sab 'label=...' --out OUT.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens import adapters as A, common as C  # noqa: E402
from score_regions import auc  # noqa: E402


def box_mask(boxes, H, W):
    m = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in boxes:
        m[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = True
    return m


def zmap(mapdir, models, tag, depths, sc, H, W):
    """mean over depths of the mean over members of per-member z (z over the scored pixels pooled over depths)."""
    Es = []
    for m in models:
        vals = [A.to_unit(m, tifffile.imread(Path(mapdir) / 'maps' / m / f'{m}__reverse__{tag}__d{d:+03d}.tif'))
                for d in depths]
        allv = np.concatenate([v[sc] for v in vals]).astype(np.float64)
        mu, sd = allv.mean(), allv.std() + 1e-6
        Es.append(np.mean([(v - mu) / sd for v in vals], 0))
    return np.mean(Es, 0).astype(np.float32)


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = rng.choice(d, (n, d.size)).mean(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def score(E, reg, masks, H, W):
    out = {}
    Lm, Im, Rm, Sm, Sg = (masks[k] for k in ('L', 'I', 'R', 'S', 'Sg'))
    for name, Nm in (('I', Im), ('R', Rm), ('S', Sm), ('Sg', Sg)):
        li, nu = E[Lm], E[Nm]
        out[name] = dict(letter_mean=round(float(li.mean()), 3), null_mean=round(float(nu.mean()), 3),
                         adv=round(float(li.mean() - nu.mean()), 3), auc=round(auc(li, nu), 3), null_px=int(nu.size))
    # per-letter paired differences vs own interline boxes and own ring
    ok = masks['ok']
    lm16 = masks['lm16']; lm8 = masks['lm8']
    per = dict(I=[], R=[])
    for (r0, r1, c0, c1) in reg['letters_raw']:
        box = np.zeros((H, W), bool); box[r0:r1, c0:c1] = True
        b = box & ok
        if b.sum() < 500:
            continue
        own_i = np.zeros((H, W), bool)
        h = r1 - r0
        for rr0, rr1 in ((r1 + 16, r1 + 16 + h // 2), (r0 - 16 - h // 2, r0 - 16)):
            own_i[max(0, rr0):max(0, rr1), c0:c1] = True
        own_i &= ~lm16 & ok
        ring = np.zeros((H, W), bool); ring[max(0, r0 - 40):r1 + 40, max(0, c0 - 40):c1 + 40] = True
        ring &= ~lm8 & ok
        if own_i.sum() >= 500:
            per['I'].append(float(E[b].mean() - E[own_i].mean()))
        if ring.sum() >= 500:
            per['R'].append(float(E[b].mean() - E[ring].mean()))
    for k in per:
        d = per[k]
        out[f'per_letter_{k}'] = dict(n=len(d), mean=round(float(np.mean(d)), 3) if d else None,
                                      ci95=boot(d) if len(d) > 2 else None, positive=int(np.sum(np.array(d) > 0)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render', required=True); ap.add_argument('--regions', required=True)
    ap.add_argument('--ctx', required=True); ap.add_argument('--need', required=True)
    ap.add_argument('--real', action='append', required=True); ap.add_argument('--sab', action='append', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = json.loads(Path(a.regions).read_text())
    arr, at = C.open_render(a.render)
    n, H, W = arr.shape
    valid = C.valid_mask_all_layers(arr, k_range=at['ensemble_render'].get('rendered_k'))
    ok = valid & np.load(a.ctx)
    sc = np.load(a.need) & valid
    lm = box_mask(reg['letters_raw'], H, W)
    lm8 = ndimage.binary_dilation(lm, iterations=8); lm16 = ndimage.binary_dilation(lm, iterations=16)
    Sm = box_mask(reg['shift_plus'] + reg['shift_minus'], H, W)
    Im = box_mask(reg['interline'], H, W) & ~lm16
    Rm = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in reg['letters_raw']:
        Rm[max(0, r0 - 40):r1 + 40, max(0, c0 - 40):c1 + 40] = True
    Rm &= ~lm8
    masks = dict(L=lm & ok, I=Im & ok, R=Rm & ok, S=Sm & ok, Sg=Sm & ~lm8 & ok, ok=ok, lm8=lm8, lm16=lm16)
    res = dict(rule=__doc__.split('usage:')[0], px={k: int(v.sum()) for k, v in masks.items() if k not in ('lm8', 'lm16')},
               real={}, sab={})
    for kind, specs in (('real', a.real), ('sab', a.sab)):
        for spec in specs:
            lab, rest = spec.split('=', 1)
            mapdir, models, tag, depths = rest.split(':', 3)
            E = zmap(mapdir, models.split(','), tag, C.parse_depths(depths), sc, H, W)
            res[kind][lab] = score(E, reg, masks, H, W)
            print(kind, lab, json.dumps(res[kind][lab]), flush=True)
    # verdict
    res['verdict'] = {}
    for lab, r in res['real'].items():
        s = res['sab'].get(lab)
        v = {}
        for k in ('I', 'R'):
            pr = r[f'per_letter_{k}']
            real_ok = pr['ci95'] is not None and pr['ci95'][0] > 0
            sab_ok = True
            if s is not None:
                ps = s[f'per_letter_{k}']
                sab_ok = (ps['mean'] < 0.25 * pr['mean']) or (ps['ci95'] is not None and ps['ci95'][0] <= 0)
            v[k] = dict(real_ci_above_0=real_ok, sabotage_fails=sab_ok)
        v['INK_EVIDENCED'] = bool(all(v[k]['real_ci_above_0'] and v[k]['sabotage_fails'] for k in ('I', 'R'))
                                  and s is not None)
        res['verdict'][lab] = v
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res['verdict'], indent=1))


if __name__ == '__main__':
    main()
