"""AUC vs planted normal offset of the organiser surface (direct canvas scoring; the planted-shift correspondence is
pixel-to-pixel: tangential slip measured 0). Label-free depth order per arm; also pooled over w058+w060."""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
from vlib import metrics as M
arms = [('M10', -10), ('M6', -6), ('M4', -4), ('M2', -2), ('R', 0), ('P2', 2), ('P4', 4), ('P6', 6), ('P10', 10)]
res = {}; pooled = {}
for w in G.WINDS:
    g = np.load(G.W / 'org' / w / 'geo.npz'); band = g['band']; lab = g['label'].astype(bool) & band
    for name, off in arms:
        if (G.ARMS / f'{w}_{name}_pred_reverse.tif').exists():
            pf, pr = G.load_pair(f'{w}_{name}')          # GPU fp16
        elif (G.ARMS / f'{w}_{name}c_pred_reverse.tif').exists():
            pf, pr = G.load_pair(f'{w}_{name}c')         # CPU fp32 (measured identical to 4 decimals of AUC)
        else:
            continue
        r, p = G.score_canvas(pf, pr, lab, band, c2=False)
        res.setdefault(name, {})[w] = dict(off=off, auc=r['auc'], dir=r['chosen'], auc_fwd=r['auc_fwd'], auc_rev=r['auc_rev'],
                                            elong=r['elong'])
        pooled.setdefault(name, []).append(M.block_hists(M.to_u8(p), lab, band))
for name, off in arms:
    if name in pooled and len(pooled[name]) == 2:
        h = (np.concatenate([x[0] for x in pooled[name]]), np.concatenate([x[1] for x in pooled[name]]))
        res[name]['pooled'] = M.auc_from(h[0].sum(0), h[1].sum(0))
        print('%-4s off %+3d  w058 %.4f (%s)  w060 %.4f (%s)  pooled %.4f' % (name, off, res[name]['w058']['auc'], res[name]['w058']['dir'][0],
              res[name]['w060']['auc'], res[name]['w060']['dir'][0], res[name]['pooled']))
json.dump(res, open(f'{_P.OUT}/g1g2/dose_response.json', 'w'), indent=1, default=float)
