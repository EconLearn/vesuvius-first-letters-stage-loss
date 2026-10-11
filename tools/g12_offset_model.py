"""Offset-only model of an automatic surface's ink loss.
For each organiser label pixel q: s(q) = signed offset of the nearest automatic point along the organiser normal.
Composite prediction = organiser dose canvases (planted offsets) linearly interpolated at s(q); d(q) > 14 -> 0
(same coverage rule as e2e@14). AUC(composite) = what the auto surface would score if its ONLY defect were the
per-pixel normal offset. Gap to the actual auto AUC = tilt / waviness / in-plane distortion / flatten.
usage: g12_offset_model.py TAG"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
from vlib import metrics as M
tag = sys.argv[1]
OFF = [(-10, 'M10'), (-6, 'M6'), (-4, 'M4'), (-2, 'M2'), (0, 'R'), (2, 'P2'), (4, 'P4'), (6, 'P6'), (10, 'P10')]
meta = json.loads((G.ARMS / f'auto_{tag}.json').read_text())
P = np.concatenate([np.load(G.ARMS / f'{c["name"]}_pts.npy').reshape(-1, 3) for c in meta['comps']])
P = P[np.isfinite(P).all(1)]
T = cKDTree(P, balanced_tree=False, compact_nodes=False)
out = {}; pooled = {'composite': [], 'composite_noclip': []}
for w in G.WINDS:
    g = np.load(G.W / 'org' / w / 'geo.npz'); band = g['band']; lab = g['label'].astype(bool) & band
    Q = g['pts'][band]; nQ = g['nrm'][band]
    d, j = T.query(Q, distance_upper_bound=60, workers=4); hit = np.isfinite(d); jf = np.where(hit, j, 0)
    s = ((P[jf] - Q) * nQ).sum(1)
    offs, canv = [], []
    for o, name in OFF:
        for nm in (f'{w}_{name}', f'{w}_{name}c'):
            if (G.ARMS / f'{nm}_pred_reverse.tif').exists():
                offs.append(o); canv.append(G.load_pair(nm)[1][band]); break
    offs = np.array(offs); C = np.stack(canv)          # (k, n) reverse-order predictions at the band pixels
    sc = np.clip(s, offs.min(), offs.max())
    hi = np.clip(np.searchsorted(offs, sc), 1, len(offs) - 1); lo = hi - 1
    t = (sc - offs[lo]) / (offs[hi] - offs[lo])
    idx = np.arange(len(sc))
    comp = (1 - t) * C[lo, idx] + t * C[hi, idx]
    cov = hit & (d <= G.RCOV)
    canvas = np.zeros(band.shape, np.float32); canvas[band] = np.where(cov, comp, 0)
    canvas2 = np.zeros(band.shape, np.float32); canvas2[band] = comp
    a = M.auc(canvas, lab, band); a2 = M.auc(canvas2, lab, band)
    pooled['composite'].append(M.block_hists(M.to_u8(canvas), lab, band)); pooled['composite_noclip'].append(M.block_hists(M.to_u8(canvas2), lab, band))
    out[w] = dict(offsets_available=offs.tolist(), auc_offset_only_e2e14=a, auc_offset_only_nocap=a2, coverage14=float(cov.mean()),
                  signed_offset_pct=[float(x) for x in np.percentile(s[cov], [5, 25, 50, 75, 95])])
    print(tag, w, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 4)) for k, v in out[w].items()}), flush=True)
for k, hs in pooled.items():
    h = (np.concatenate([x[0] for x in hs]), np.concatenate([x[1] for x in hs]))
    out[f'pooled_{k}'] = M.auc_from(h[0].sum(0), h[1].sum(0))
print(tag, 'pooled offset-only AUC (e2e@14) %.4f  (no cap) %.4f' % (out['pooled_composite'], out['pooled_composite_noclip']))
json.dump(out, open(f'{_P.OUT}/g1g2/offset_model_{tag}.json', 'w'), indent=1, default=float)
