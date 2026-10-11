"""Geometry-only adherence analysis of an automatic surface vs the organiser label area (CPU).
usage: g12_geom.py TAG [TAG...]  (auto_<TAG>.json canvases)"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
out = {}
for tag in sys.argv[1:]:
    meta = json.loads((G.ARMS / f'auto_{tag}.json').read_text())
    P, Nn = [], []
    for c in meta['comps']:
        pos = np.load(G.ARMS / f'{c["name"]}_pts.npy'); nn = G._canvas_normals(pos)
        ok = np.isfinite(pos).all(-1) & np.isfinite(nn).all(-1)
        P.append(pos[ok]); Nn.append(nn[ok])
    P = np.concatenate(P); Nn = np.concatenate(Nn)
    T = cKDTree(P, balanced_tree=False, compact_nodes=False)
    for w in G.WINDS:
        g = np.load(G.W / 'org' / w / 'geo.npz'); band = g['band']
        Q = g['pts'][band]; nQ = g['nrm'][band]; lab = g['label'][band].astype(bool)
        d, j = T.query(Q, distance_upper_bound=60, workers=4)
        hit = np.isfinite(d); jf = np.where(hit, j, 0)
        s = ((P[jf] - Q) * nQ).sum(1)                      # signed offset along the organiser normal (+ = render +n side)
        tang = np.sqrt(np.maximum(d ** 2 - s ** 2, 0))
        ang = np.degrees(np.arccos(np.abs((nQ * Nn[jf]).sum(1)).clip(0, 1)))
        h14 = hit & (d <= 14)
        r = dict(n=int(len(Q)), found60=float(hit.mean()),
                 d_pct=[float(x) for x in np.percentile(d[hit], [10, 25, 50, 75, 90, 95])],
                 signed_pct=[float(x) for x in np.percentile(s[h14], [5, 25, 50, 75, 95])],
                 frac_abs_signed_le=[float((h14 & (np.abs(s) <= k)).mean()) for k in (1, 2, 4, 6)],
                 tangential_median=float(np.median(tang[h14])),
                 angle_pct=[float(x) for x in np.percentile(ang[h14], [25, 50, 75, 90])],
                 ink_vs_bg_d_median=[float(np.median(d[hit & lab])), float(np.median(d[hit & ~lab]))])
        out[f'{tag}_{w}'] = r
        print(tag, w, json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r.items()}), flush=True)
json.dump(out, open(f'{_P.OUT}/g1g2/geom_{"_".join(sys.argv[1:])}.json', 'w'), indent=1)
