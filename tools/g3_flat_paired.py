#!/usr/bin/env python
"""Stitching-weight lever, paired: released v8-in maps raw vs divided by W = sum(g)/count (the weight field the
released ink8um stitching forgets to divide by). Same pixels, same mapping, paired 64-px block bootstrap.
Arms: organiser surface (our run, V8R) and automatic it30000 surface (B1), phases given on the command line;
also the organisers' released map (V8rel) with W rebuilt from the full-surface tile layout.
Sabotage: W shifted by half the stride (10 px) in both axes must NOT help (the gain must come from the exact
layout, not from smoothing).
usage: g3_flat_paired.py [PHASE ...]   (default ph0_0; every listed phase must exist for V8R, B1 and V8rel)"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import os, sys, json
import numpy as np, cv2
os.environ.setdefault('G3BAND', 'b1')
sys.path.insert(0, _P.TOOLS)
import g3_score as S
G, M = S.G, S.M
phases = sys.argv[1:] or ['ph0_0']


def canv_arm(w, g, kind, ph, mode):
    """mode: raw | flat | sab (W rolled by 10 px)"""
    band = g['band']; H, Wd = band.shape
    def fix(p, Wf):
        if mode == 'raw':
            return p
        if mode == 'sab':
            Wf = np.roll(Wf, (10, 10), axis=(0, 1))
        return p / np.maximum(Wf, 1e-3)
    if kind == 'V8R':
        p = S.tifffile.imread(S.V8 / f'{w}_R__rev__{ph}.tif').astype(np.float32) / 65535
        return fix(p, S.wfield(G.ARMS / f'{w}_R.zarr', f'{w}_R', ph))
    if kind == 'V8rel':
        r0, r1, c0, c1 = g['sub']
        p = cv2.imread(str(G.E1 / f'1447_{w}' / 'v8in_released.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)[r0:r1, c0:c1] / 255
        return fix(p, S.wfield_released(w)[r0:r1, c0:c1])
    A = ARM
    Q = g['pts'][band]
    d, j = A.corr(w, Q)
    hit = np.isfinite(d) & (d <= S.RCOV)
    idx = A.IDX[np.where(hit, j, 0)]
    vals = np.zeros(len(Q), np.float32)
    for k, n in enumerate(A.names):
        sel = hit & (idx[:, 0] == k)
        if sel.any():
            p = S.tifffile.imread(S.V8 / f'{n}__fwd__{ph}.tif').astype(np.float32) / 65535
            p = fix(p, S.wfield(G.ARMS / f'{n}.zarr', n, ph))
            vals[sel] = p[idx[sel, 1], idx[sel, 2]]
    C = np.zeros((H, Wd), np.float32); C[band] = vals
    return C


ARM = S.AutoArm(G.ARMS, 'it30000flat')
res = {}
for ph in phases:
    for kind in ('V8R', 'B1', 'V8rel'):
        if kind == 'V8rel' and ph != 'ph0_0':
            continue
        hs = {m: [] for m in ('raw', 'flat', 'sab')}
        for w in G.WINDS:
            g = np.load(G.W / 'org' / w / 'geo.npz')
            band = g['band']; label = g['label'].astype(bool) & band
            for m in hs:
                p = canv_arm(w, g, kind, ph, m)
                p = np.where(band, p, 0); p = p / max(float(np.percentile(p[band], 99.9)), 1e-6)
                hs[m].append(M.block_hists(M.to_u8(np.clip(p, 0, 1)), label, band))
        cat = {m: (np.concatenate([h[0] for h in v]), np.concatenate([h[1] for h in v])) for m, v in hs.items()}
        auc = {m: M.auc_from(c[0].sum(0), c[1].sum(0)) for m, c in cat.items()}
        r = dict(auc=auc, d_flat=auc['flat'] - auc['raw'], ci_flat=G.paired_ci(cat['raw'], cat['flat']),
                 d_sab=auc['sab'] - auc['raw'], ci_sab=G.paired_ci(cat['raw'], cat['sab']))
        res[f'{kind}_{ph}'] = r
        print(f'{kind:6s} {ph:8s} raw {auc["raw"]:.4f}  flat {auc["flat"]:.4f}  d {r["d_flat"]:+.4f} [{r["ci_flat"][0]:+.4f}, {r["ci_flat"][1]:+.4f}]'
              f'   sabotage(W shifted 10 px) d {r["d_sab"]:+.4f} [{r["ci_sab"][0]:+.4f}, {r["ci_sab"][1]:+.4f}]', flush=True)
json.dump(res, open(f'{_P.OUT}/g3/flat_paired.json', 'w'), indent=1, default=float)
