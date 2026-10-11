#!/usr/bin/env python
"""Winding-consistency metric for an automatic (fitted) surface, measured against human-made reference surfaces.

WHY: nearest-point adherence ("share of reference pixels within k vox of the fitted surface") cannot see a fitted
sheet that hops between true sheets: in a crushed scroll some true sheet is always within half the pitch (~14 vox
on PHerc1447). G3 (2026-10-08) found the official 30k-step spiral fit of PHerc1447 z 13100-14100 sits within 6 vox of
the organiser surface over ~80% of the area, yet scatters one winding's text block over 16 separate stretches of its
own flattened grid. Readers see the fitted surface in its own flattened frame, so this fragmentation, not adherence,
decides legibility. This module measures it.

WHAT (all from reference->candidate correspondences; the candidate is never modified):
  For reference samples q (every `stride` px of the reference render grid inside a z band, optionally restricted to a
  mask such as the ink labels), find the candidate surface point "on the same sheet":
    k nearest candidate grid vertices (default k=12) -> keep those whose tangential distance to q is <= 1.5 grid steps
    -> pick the one with the smallest |normal distance| (vertex normals from the grid) -> accept if <= cap (default
    14 vox = half the 1447 sheet pitch). Its grid (row, col) / scale gives the candidate pixel (r, c).
  Reported (per reference surface, and pooled):
    matched_frac           accepted / samples
    densest_share          largest share of the MASKED samples falling inside one candidate-frame window of
                           `window_px` columns (default 4,000 px) -- 1.0 = the whole masked region (e.g. a text block)
                           lies in one contiguous stretch of the candidate's own frame
    n_stretches            number of window_px-wide column bins holding >= 1% of the masked samples
    switch_rate_per_300px  SECONDARY (sensitive to cap_switch, see below). For sample pairs `step_px` apart along the
                           reference x and z axes, both matched within cap_switch (default 3 vox, i.e. the candidate
                           really lies on the sheet there): a pair "switches" when
                           its candidate positions are > jump_px apart (default 500 px) although the reference pair is
                           only step_px apart; converted to the probability of >= 1 switch over 300 px
                           (1 - (1 - p)^(300 / step_px)); reported for x, z and their mean
    adherence_le2 / le6    nearest-surface share within 2 / 6 vox (the old instrument, for comparison)
  PRIMARY metrics: densest_share and n_stretches (cap-robust). Also report on_sheet_le3 = share of band samples whose
  candidate lies within 3 vox (adherence_le2/le6 are the old nearest-surface numbers).
  Switch-rate sensitivity measured on the 1447 it30000 fit (w058, certainty-mask region, p per 8-px step x/z):
    cap_switch 14: 0.057/0.057   6: 0.043/0.037   3: 0.0018/0.0011  -- loose caps let samples that sit between two
    candidate sheets flip between them, so compare surfaces only at a fixed cap_switch.
  Lower switch rate and higher densest_share are better. A surface that is consistent everywhere scores
  switch_rate 0 and densest_share 1 (for a masked region narrower than window_px).

INTERFACE
  from winding_metric import winding_metric
  res = winding_metric(cand_tifxyz, refs, zband=(13150, 14050), mask='labels', stride=4, cap=14.0,
                       window_px=4000, step_px=8, jump_px=500, cap_switch=3.0)
    cand_tifxyz : path to the candidate surface (VC tifxyz: x.tif, y.tif, z.tif, meta.json with 'scale')
    refs        : list of dicts {name, tifxyz, labels(optional png, organiser frame)} or the shortcut names
                  'w058', 'w060', 'w062' (PHerc1447 organiser surfaces, local HF copy)
    mask        : 'labels' (ink-label pixels; the default, measures text fragmentation), 'band' (every reference pixel
                  in the z band; measures surface-wide consistency) or a callable(ref_name, rows, cols) -> bool array
  returns a dict (JSON-serialisable); CLI writes it to --out.
  CLI:  python winding_metric.py CAND.tifxyz --ref w058 --ref w060 --zband 13150 14050 --mask labels --out res.json
        python winding_metric.py --selftest      (identity must score 0 switches / share 1; sabotage must not)
Known-answer values (G3, measured on the official it30000 fit of PHerc1447 z 13100-14100, labels mask):
  w058 densest_share 0.27, n_stretches 16; w060 0.41, 5 (pixel-level correspondences of the scorer; this module's
  vertex-projection numbers are expected to agree within a few points -- see the selftest output).
"""
from __future__ import annotations
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

HF = Path(f'{_P.DATA}/hf/ink-8um-pherc1447-surfaces')


def load_tifxyz(d):
    import tifffile
    d = Path(d)
    x = tifffile.imread(d / 'x.tif').astype(np.float32); y = tifffile.imread(d / 'y.tif').astype(np.float32)
    z = tifffile.imread(d / 'z.tif').astype(np.float32)
    meta = json.loads((d / 'meta.json').read_text())
    P = np.stack([x, y, z], -1)
    valid = (x != -1) & np.isfinite(x) & ~((x == 0) & (y == 0) & (z == 0))
    P[~valid] = np.nan
    return P, valid, float(meta['scale'][0])


def grid_normals(P):
    N = np.full(P.shape, np.nan, np.float32)
    N[1:-1, 1:-1] = np.cross(P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1])
    return N / np.linalg.norm(N, axis=-1, keepdims=True)


def _ref(r):
    if isinstance(r, str):
        return dict(name=r, tifxyz=HF / r / 'tifxyz', labels=(HF / r / 'labels' / 'inklabels.png') if (HF / r / 'labels').exists() else None)
    return r


class Candidate:
    def __init__(self, tifxyz):
        self.P, self.valid, self.s = load_tifxyz(tifxyz)
        self.N = grid_normals(self.P)
        ok = self.valid & np.isfinite(self.N).all(-1)
        self.vi = np.argwhere(ok); self.V = self.P[ok]; self.VN = self.N[ok]
        self.T = cKDTree(self.V)
        self.step = float(np.nanmedian(np.linalg.norm(self.P[:, 1:] - self.P[:, :-1], axis=-1)))   # vox per grid step

    def project(self, q, cap=14.0, k=12):
        """candidate pixel (row, col) in the candidate's own render frame for points q (n, 3); NaN where none."""
        d, j = self.T.query(q, k=k, workers=8)
        j = np.minimum(j, len(self.V) - 1)
        diff = q[:, None, :] - self.V[j]                       # (n, k, 3)
        nd = np.einsum('nkc,nkc->nk', diff, self.VN[j])         # signed normal distance
        tang = np.sqrt(np.maximum((diff ** 2).sum(-1) - nd ** 2, 0))
        score = np.where(np.isfinite(d) & (tang <= 1.5 * self.step), np.abs(nd), np.inf)
        b = np.argmin(score, 1)
        best = score[np.arange(len(q)), b]
        jj = j[np.arange(len(q)), b]
        rc = self.vi[jj].astype(np.float64) / self.s          # grid -> candidate render pixels
        rc[~(best <= cap)] = np.nan
        return rc, np.where(best <= cap, best, np.nan)


def winding_metric(cand_tifxyz, refs, zband=(13150, 14050), mask='labels', stride=4, cap=14.0, window_px=4000,
                   step_px=8, jump_px=500, k=12, cand=None, cap_switch=3.0, switch_region='band'):
    import cv2
    C = cand or Candidate(cand_tifxyz)
    out = dict(candidate=str(cand_tifxyz), zband=list(zband), mask=mask if isinstance(mask, str) else 'custom', stride=stride,
               cap=cap, window_px=window_px, step_px=step_px, jump_px=jump_px, cap_switch=cap_switch,
               switch_region=switch_region, refs={})
    pooled_cols, pooled_sw = [], []
    for r in map(_ref, refs):
        P, valid, s = load_tifxyz(r['tifxyz'])
        H, W = int(P.shape[0] / s), int(P.shape[1] / s)
        rows = np.arange(0, H, stride); cols = np.arange(0, W, stride)
        # reference points at render pixels (bilinear on the grid, pixel-centre convention as vc_render_tifxyz)
        gr = (rows + 0.5) * s; gc = (cols + 0.5) * s
        GR, GC = np.meshgrid(gr, gc, indexing='ij')
        r0 = np.clip(np.floor(GR).astype(int), 0, P.shape[0] - 2); c0 = np.clip(np.floor(GC).astype(int), 0, P.shape[1] - 2)
        fr = (GR - r0)[..., None]; fc = (GC - c0)[..., None]
        Q = (P[r0, c0] * (1 - fr) * (1 - fc) + P[r0, c0 + 1] * (1 - fr) * fc + P[r0 + 1, c0] * fr * (1 - fc) + P[r0 + 1, c0 + 1] * fr * fc)
        z = Q[..., 2]
        inb = np.isfinite(Q).all(-1) & (z >= zband[0]) & (z < zband[1])
        if callable(mask):
            m = mask(r['name'], rows, cols) & inb
        elif mask == 'labels':
            lab = cv2.imread(str(r['labels']), cv2.IMREAD_GRAYSCALE) > 127
            m = lab[np.ix_(np.clip(rows, 0, lab.shape[0] - 1), np.clip(cols, 0, lab.shape[1] - 1))] & inb
        else:
            m = inb
        RC = np.full(Q.shape[:2] + (2,), np.nan); ND = np.full(Q.shape[:2], np.nan)
        idx = np.argwhere(inb)
        rc, nd = C.project(Q[inb], cap=cap, k=k)
        RC[idx[:, 0], idx[:, 1]] = rc; ND[idx[:, 0], idx[:, 1]] = nd
        matched = np.isfinite(ND) & inb
        # fragmentation of the masked region in the candidate frame
        mc = RC[m & matched][:, 1]
        res = dict(samples=int(inb.sum()), masked=int(m.sum()), matched_frac=float(matched[inb].mean()) if inb.any() else None)
        if len(mc):
            bins = np.arange(np.floor(mc.min()), mc.max() + window_px, window_px)
            hist = np.histogram(mc, bins=bins)[0]
            # densest window: slide in 1/10-window steps
            fine = np.histogram(mc, bins=np.arange(np.floor(mc.min()), mc.max() + window_px / 10, window_px / 10))[0]
            dens = np.convolve(fine, np.ones(10, int), 'valid').max() if len(fine) >= 10 else fine.sum()
            res.update(densest_share=float(dens / len(mc)), n_stretches=int((hist >= 0.01 * len(mc)).sum()),
                       masked_matched=int(len(mc)))
        # switch rate (all matched band samples, both axes)
        st = max(1, step_px // stride)
        sw = {}
        conf = np.isfinite(ND) & (ND <= cap_switch)            # only pairs where the candidate is ON the sheet
        if switch_region == 'certainty' and (Path(r['tifxyz']).parent / 'labels' / 'mask.png').exists():
            cm = cv2.imread(str(Path(r['tifxyz']).parent / 'labels' / 'mask.png'), cv2.IMREAD_GRAYSCALE) > 127
            conf &= cm[np.ix_(np.clip(rows, 0, cm.shape[0] - 1), np.clip(cols, 0, cm.shape[1] - 1))]
        elif switch_region == 'mask':
            conf &= m
        for ax, (a, b, ca, cb) in (('x', (RC[:, st:], RC[:, :-st], conf[:, st:], conf[:, :-st])),
                                   ('z', (RC[st:, :], RC[:-st, :], conf[st:, :], conf[:-st, :]))):
            dd = np.linalg.norm(a - b, axis=-1)
            ok = np.isfinite(dd) & ca & cb
            p = float((dd[ok] > jump_px).mean()) if ok.any() else np.nan
            sw[ax] = dict(p_per_step=p, per_300px=float(1 - (1 - p) ** (300 / (st * stride))) if np.isfinite(p) else None, pairs=int(ok.sum()))
            pooled_sw.append((dd[ok] > jump_px))
        res['switch'] = sw
        res['switch_rate_per_300px_mean'] = float(np.mean([sw['x']['per_300px'], sw['z']['per_300px']]))
        nd_ok = ND[inb & np.isfinite(ND)]
        res['on_sheet_le3'] = float((ND[inb] <= 3).mean()) if inb.any() else None
        res['adherence_le2'] = float((ND[inb] <= 2).mean()) if inb.any() else None
        res['adherence_le6'] = float((ND[inb] <= 6).mean()) if inb.any() else None
        out['refs'][r['name']] = res
        pooled_cols.append(res)
    allsw = np.concatenate(pooled_sw) if pooled_sw else np.zeros(0, bool)
    p = float(allsw.mean()) if len(allsw) else np.nan
    out['pooled'] = dict(switch_p_per_step=p, switch_rate_per_300px=float(1 - (1 - p) ** (300 / step_px)) if np.isfinite(p) else None,
                         densest_share_mean=float(np.mean([r['densest_share'] for r in pooled_cols if 'densest_share' in r])) if pooled_cols else None)
    return out


def selftest():
    """identity: the organiser surface measured against itself must give switch ~0 and densest_share ~1;
    sabotage: the same surface with 500-px column blocks shuffled must give many switches and a low share."""
    import tempfile, shutil, tifffile
    src = HF / 'w058' / 'tifxyz'
    tmp = Path(tempfile.mkdtemp())
    try:
        P, valid, s = load_tifxyz(src)
        rng = np.random.default_rng(0)
        blk = int(500 * s)                                     # 500 px in grid columns
        nb = P.shape[1] // blk
        perm = rng.permutation(nb)
        Ps = np.concatenate([P[:, i * blk:(i + 1) * blk] for i in perm] + [P[:, nb * blk:]], 1)
        d = tmp / 'shuf.tifxyz'; d.mkdir()
        for c, nm in enumerate('xyz'):
            tifffile.imwrite(d / f'{nm}.tif', np.where(np.isfinite(Ps[..., c]), Ps[..., c], -1).astype(np.float32))
        (d / 'meta.json').write_text((src / 'meta.json').read_text())
        a = winding_metric(src, ['w058'], stride=8)
        b = winding_metric(d, ['w058'], stride=8)
        ra, rb = a['refs']['w058'], b['refs']['w058']
        print('IDENTITY  densest_share %.3f  n_stretches %d  switch/300px %.4f  matched %.3f' % (ra['densest_share'], ra['n_stretches'], ra['switch_rate_per_300px_mean'], ra['matched_frac']))
        print('SABOTAGE  densest_share %.3f  n_stretches %d  switch/300px %.4f  matched %.3f' % (rb['densest_share'], rb['n_stretches'], rb['switch_rate_per_300px_mean'], rb['matched_frac']))
        ok = ra['switch_rate_per_300px_mean'] < 0.02 and ra['densest_share'] > 0.95 and rb['switch_rate_per_300px_mean'] > 0.1
        print('SELFTEST', 'PASS' if ok else 'FAIL')
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cand', nargs='?'); ap.add_argument('--ref', action='append', default=[])
    ap.add_argument('--zband', type=float, nargs=2, default=[13150, 14050]); ap.add_argument('--mask', default='labels')
    ap.add_argument('--stride', type=int, default=4); ap.add_argument('--cap', type=float, default=14.0)
    ap.add_argument('--window-px', type=int, default=4000); ap.add_argument('--step-px', type=int, default=8)
    ap.add_argument('--jump-px', type=int, default=500); ap.add_argument('--cap-switch', type=float, default=3.0)
    ap.add_argument('--switch-region', default='band', choices=['band', 'certainty', 'mask'])
    ap.add_argument('--out'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    res = winding_metric(a.cand, a.ref or ['w058', 'w060'], tuple(a.zband), a.mask, a.stride, a.cap, a.window_px, a.step_px,
                         a.jump_px, cap_switch=a.cap_switch, switch_region=a.switch_region)
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == '__main__':
    main()
