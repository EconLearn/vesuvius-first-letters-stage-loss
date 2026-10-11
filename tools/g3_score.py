#!/usr/bin/env python
"""G3 scorer: v8-in arms on the G1G2 bench (PHerc1447 w058+w060, z 13150-14050, certainty mask ∩ band).

Arms (whatever predictions exist are scored; see SUBMISSION.md):
  V8rel            organisers' released v8-in map on their render (E1 copy), cropped like G1G2
  V8R_<ph>         our v8-in on our render of the organiser surface (<w>_R), order rev, direct (same canvas)
  <ARM>_<ph>       automatic arms, mapped by 3D nearest-surface correspondence, e2e cap 14 vox (uncovered = 0)
  ..._avg5         mean of the 5 tiling-phase probability maps in canvas space (before mapping)
AUTO arm table: name -> (arm dir, tag, order). CIs: paired 64-px block bootstrap (1,000), pooled over windings.
usage: g3_score.py [--c2] [--save-maps] [--only ARM ...]   -> $VC_OUT/g3/g3_scores_<band>[_flat]_<arms>.json
env: G3BAND=b1|b2 (bench), G3FLAT=1 (divide v8-in maps by the stitching weight field W)
"""
from __future__ import annotations
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import sys, json, time, argparse
from pathlib import Path
import numpy as np, tifffile
from scipy.spatial import cKDTree
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G  # noqa
from vlib import metrics as M  # noqa
import os
BAND = os.environ.get('G3BAND', 'b1'); PFX = ''
if BAND == 'b2':   # band-2 replication bench (w060, z 11050-12300), G1G2 globals re-pointed as in g3_band2.py
    G.W = Path(f'{_P.WORK}/g3/b2'); G.ARMS = G.W / 'arms'; G.ZLO, G.ZHI = 11050, 12300; G.WINDS = ['w060']; PFX = 'b2'

V8 = Path(f'{_P.WORK}/g3/v8'); OUT3 = Path(f'{_P.OUT}/g3'); ARMS3 = Path(f'{_P.WORK}/g3/arms')
MAPS = Path(f'{_P.WORK}/g3/maps')
PH = [(0, 0), (5, 5), (10, 10), (16, 16), (10, 0)]
PHN = [f'ph{a}_{b}' for a, b in PH]
AUTO = {   # arm -> (dir, tag, order)
    'B1': (G.ARMS, 'it30000flat', 'fwd'),
    'B1b2': (Path(f'{_P.WORK}/g3/b2/arms'), 'b2it30000flat', 'fwd'),
    'B1rev': (G.ARMS, 'it30000flat', 'rev'),
    'S1': (ARMS3, 'it30000snap', 'fwd'),
    'SR': (ARMS3, 'it30000snapSR', 'fwd'),
    'SL': (ARMS3, 'it30000snapSL', 'fwd'),
}
WINDS = G.WINDS; RCOV = G.RCOV; RMAX = G.RMAX


FLAT = os.environ.get('G3FLAT') == '1'   # divide v8-in maps by the released stitching's weight field (lever test)
WF = Path(f'{_P.WORK}/g3/wfield')


def wfield(zpath, key, ph='ph0_0'):
    """W = sum(g) / count of the released ink8um stitching for this canvas's tile layout (released ink8um predict_stack)."""
    f = WF / f'{key}__{ph}.npy'
    if f.exists():
        return np.load(f)
    import zarr
    sys.path.insert(0, _P.V8IN_DIR)
    from ink8um import inference as I8
    a = np.asarray(zarr.open(str(zpath), mode='r')['0'])
    st = np.clip(a[2:26] if a.shape[0] == 28 else a, 0, 200).astype(np.uint8).transpose(1, 2, 0)
    dr, dc = [int(x) for x in ph[2:].split('_')]
    if dr or dc:
        st = np.pad(st, ((dr, 0), (dc, 0), (0, 0)))
    g = I8.gkern(64, 1); g = (g / g.max()).astype(np.float32)
    num = np.zeros(st.shape[:2], np.float32); cnt = np.zeros_like(num)
    for x, y in I8.tile_positions(I8.coverage_mask(st), 64, 21):
        num[y:y + 64, x:x + 64] += g; cnt[y:y + 64, x:x + 64] += 1
    W = np.divide(num, cnt, out=np.ones_like(num), where=cnt > 0)[dr:, dc:]
    WF.mkdir(parents=True, exist_ok=True); np.save(f, W.astype(np.float32))
    return W


def wfield_released(w):
    """W for the organisers' released full-surface v8-in map (tile grid anchored at the full surface's (0, 0)),
    cropped to the E1 frame."""
    f = WF / f'released_{w}.npy'
    if f.exists():
        return np.load(f)
    import glob
    sys.path.insert(0, _P.V8IN_DIR)
    from ink8um import inference as I8
    fs = sorted(glob.glob(f'{_P.DATA}/hf/ink-8um-pherc1447-surfaces/{w}/layers/*.tif'))
    cov = None
    for fl in fs:
        im = tifffile.imread(fl)
        cov = (im > 0) if cov is None else (cov | (im > 0))
    covm = I8.coverage_mask(cov[..., None].astype(np.uint8))
    g = I8.gkern(64, 1); g = (g / g.max()).astype(np.float32)
    num = np.zeros(cov.shape, np.float32); cnt = np.zeros_like(num)
    for x, y in I8.tile_positions(covm, 64, 21):
        num[y:y + 64, x:x + 64] += g; cnt[y:y + 64, x:x + 64] += 1
    W = np.divide(num, cnt, out=np.ones_like(num), where=cnt > 0)
    i0, i1, j0, j1 = json.loads((G.E1 / f'1447_{w}' / 'prep.json').read_text())['meta']['crop']
    W = W[i0:i1, j0:j1]
    WF.mkdir(parents=True, exist_ok=True); np.save(f, W.astype(np.float32))
    return W


def rd(p, zpath=None, key=None, ph='ph0_0'):
    a = tifffile.imread(p).astype(np.float32) / 65535.0
    if FLAT and zpath is not None:
        a = a / np.maximum(wfield(zpath, key, ph), 1e-3)
        a = a / max(float(np.percentile(a[a > 0], 99.9)) if (a > 0).any() else 1.0, 1e-6)   # back to ~[0, 1]
        a = np.clip(a, 0, 1)
    return a


def phase_sets(canv_names, order):
    """which phases exist for ALL canvases of an arm."""
    have = []
    for ph in PHN:
        if all((V8 / f'{n}__{order}__{ph}.tif').exists() for n in canv_names):
            have.append(ph)
    return have


class AutoArm:
    def __init__(self, d, tag):
        self.dir = Path(d)
        meta = json.loads((Path(d) / f'auto_{tag}.json').read_text())
        self.names = [c['name'] for c in meta['comps']]
        P, IDX = [], []
        for k, c in enumerate(meta['comps']):
            pos = np.load(Path(d) / f'{c["name"]}_pts.npy')
            ok = np.isfinite(pos).all(-1)          # same validity as the NEED masks (g3_need.py)
            P.append(pos[ok]); IDX.append(np.column_stack([np.full(int(ok.sum()), k), np.argwhere(ok)]))
        self.P = np.concatenate(P); self.IDX = np.concatenate(IDX)
        self.T = cKDTree(self.P, balanced_tree=False, compact_nodes=False)
        self.map = {}

    def corr(self, w, Q):
        if w not in self.map:
            d, j = self.T.query(Q, distance_upper_bound=RMAX, workers=8)
            self.map[w] = (d, j)
        return self.map[w]

    def values(self, w, Q, order, phases):
        d, j = self.corr(w, Q)
        hit = np.isfinite(d) & (d <= RCOV)
        out = np.zeros(len(Q), np.float32)
        acc = None
        for ph in phases:
            vals = np.zeros(len(Q), np.float32)
            idx = self.IDX[np.where(hit, j, 0)]
            for k, n in enumerate(self.names):
                sel = hit & (idx[:, 0] == k)
                if sel.any():
                    p = rd(V8 / f'{n}__{order}__{ph}.tif', self.dir / f'{n}.zarr', n, ph)
                    vals[sel] = p[idx[sel, 1], idx[sel, 2]]
            acc = vals if acc is None else acc + vals
        out[hit] = (acc / len(phases))[hit]
        return out, d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--c2', action='store_true'); ap.add_argument('--save-maps', action='store_true')
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    t0 = time.time()
    res = dict(settings=dict(band=[G.ZLO, G.ZHI], rcov=RCOV, model='v8-in released (ink8um), stride 21, 24 centred layers',
                             phases=PHN, ci='paired 64-px block bootstrap, 1000, seed 0'), windings={}, pooled={})
    hists = {}
    autos = {}
    for w in G.WINDS:
        g = np.load(G.W / 'org' / w / 'geo.npz')
        band = g['band']; label = g['label'].astype(bool) & band
        r0, r1, c0, c1 = g['sub']; H, Wd = band.shape
        Q = g['pts'][band]
        canv = {}
        v8 = G.E1 / f'1447_{w}' / 'v8in_released.png'
        import cv2
        canv['V8rel'] = cv2.imread(str(v8), cv2.IMREAD_GRAYSCALE).astype(np.float32)[r0:r1, c0:c1] / 255.0
        if FLAT:
            canv['V8rel'] = np.clip(canv['V8rel'] / np.maximum(wfield_released(w)[r0:r1, c0:c1], 1e-3), 0, None)
        # our v8-in on the organiser surface (direct)
        have = [ph for ph in PHN if (V8 / f'{PFX}{w}_R__rev__{ph}.tif').exists()]
        Rph = {ph: rd(V8 / f'{PFX}{w}_R__rev__{ph}.tif', G.ARMS / f'{w}_R.zarr', f'{PFX}{w}_R', ph) for ph in have}
        for sab in ('P10', 'M10'):
            if (V8 / f'{PFX}{w}_{sab}__rev__ph0_0.tif').exists():
                canv[f'V8{sab}_ph0_0'] = rd(V8 / f'{PFX}{w}_{sab}__rev__ph0_0.tif')
        for ph in have:
            canv[f'V8R_{ph}'] = Rph[ph]
        if len(have) == 5:
            canv['V8R_avg5'] = np.mean([Rph[ph] for ph in PHN], 0)
        if (V8 / f'{PFX}{w}_R__fwd__ph0_0.tif').exists():
            canv['V8Rfwd_ph0_0'] = rd(V8 / f'{PFX}{w}_R__fwd__ph0_0.tif')
        # automatic arms
        adh = {}
        for arm, (d, tag, order) in AUTO.items():
            if a.only and arm not in a.only:
                continue
            if not (Path(d) / f'auto_{tag}.json').exists():
                continue
            key = (str(d), tag)
            if key not in autos:
                autos[key] = AutoArm(d, tag)
            A = autos[key]
            phs = phase_sets(A.names, order)
            if not phs:
                continue
            for ph in phs:
                v, dist = A.values(w, Q, order, [ph])
                C = np.zeros((H, Wd), np.float32); C[band] = v; canv[f'{arm}_{ph}'] = C
            if len(phs) == 5:
                v, dist = A.values(w, Q, order, phs)
                C = np.zeros((H, Wd), np.float32); C[band] = v; canv[f'{arm}_avg5'] = C
            dd = dist[np.isfinite(dist)]
            adh[arm] = dict(cov14=float((np.isfinite(dist) & (dist <= RCOV)).mean()), d_median=float(np.median(dd)))
        R = {}
        for k, p in canv.items():
            pz = np.where(band, p, 0.0).astype(np.float32)
            r = dict(auc=M.auc(pz, label, band), ci95=M.auc_ci(pz, label, band, n=300))
            r['elong'] = M.elongation(pz, label, band, G.PIX)
            base = k.split('_')[0]
            if base in adh:
                r['adherence'] = adh[base]
            if a.c2 and (k.endswith('avg5') or k.endswith('ph0_0') or k == 'V8rel'):
                r['c2_median'] = float(np.median(M.shuffle_floor(pz, label, band, seeds=8)))
            R[k] = r
            hists.setdefault(k, []).append(M.block_hists(M.to_u8(pz), label, band))
            print(w, k, 'AUC %.4f [%.4f, %.4f]  elong %.1f/%.1f (n=%d)%s' % (
                r['auc'], r['ci95'][0], r['ci95'][1], r['elong']['map'], r['elong']['labels'], r['elong']['letters'],
                ('  C2 %.3f' % r['c2_median']) if 'c2_median' in r else ''), flush=True)
            if a.save_maps and (k.endswith('avg5') or k.endswith('ph0_0') or k == 'V8rel'):
                MAPS.mkdir(parents=True, exist_ok=True)
                np.save(MAPS / f'{w}_{k}.npy', np.rint(pz * 65535).astype(np.uint16))
        res['windings'][w] = R
    # pooled + paired CIs
    pooled = {}
    for k, hs in hists.items():
        if len(hs) != len(G.WINDS):
            continue
        h = (np.concatenate([x[0] for x in hs]), np.concatenate([x[1] for x in hs]))
        pooled[k] = h
        res['pooled'][k] = dict(auc=M.auc_from(h[0].sum(0), h[1].sum(0)))
    refs = [r for r in ('V8rel', 'V8R_avg5', 'V8R_ph0_0', 'B1_ph0_0', 'B1_avg5', 'S1_ph0_0') if r in pooled]
    for k, h in pooled.items():
        for ref in refs:
            if ref == k:
                continue
            hr = pooled[ref]
            res['pooled'][k][f'd_vs_{ref}'] = res['pooled'][k]['auc'] - res['pooled'][ref]['auc']
            res['pooled'][k][f'ci_vs_{ref}'] = G.paired_ci(hr, h)
    # phase spread
    for arm in ['V8R', 'B1', 'S1', 'SR']:
        vals = [res['pooled'][f'{arm}_{ph}']['auc'] for ph in PHN if f'{arm}_{ph}' in res['pooled']]
        if len(vals) >= 2:
            res['pooled'][f'{arm}_phase_spread'] = dict(n=len(vals), mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)),
                                                       min=float(min(vals)), max=float(max(vals)))
    OUT3.mkdir(parents=True, exist_ok=True)
    tag = BAND + ('_flat' if FLAT else '') + '_' + ('_'.join(a.only) if a.only else 'all')
    (OUT3 / f'g3_scores_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print('POOLED')
    for k, r in res['pooled'].items():
        if 'auc' in r:
            extra = '  '.join(f'{x[5:]} {r[x]:+.4f} [{r["ci_" + x[2:]][0]:+.4f}, {r["ci_" + x[2:]][1]:+.4f}]'
                              for x in r if x.startswith('d_vs_'))
            print(f'{k:16s} {r["auc"]:.4f}  {extra}')
        else:
            print(k, json.dumps(r))
    print('secs', round(time.time() - t0))


if __name__ == '__main__':
    main()
