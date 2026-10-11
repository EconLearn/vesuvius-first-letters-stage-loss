"""Label-free, geometric depth order for an automatic render: sign of N_auto . N_org at matched points.
Auto render stack axis = -N_auto (flip-normals); organiser render stack axis = +N_org (no flip; read 'reverse' per the
published README). If N_auto . N_org > 0, the stacks run opposite, so auto 'forward' == organiser 'reverse'."""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
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
        d, j = T.query(g['pts'][band], distance_upper_bound=14, workers=4); hit = np.isfinite(d)
        dots = (Nn[j[hit]] * g['nrm'][band][hit]).sum(1)
        print(tag, w, 'N_auto.N_org > 0 on %.4f of %d matched px -> geometric order for auto render: %s' % (
            (dots > 0).mean(), hit.sum(), 'forward' if (dots > 0).mean() > 0.5 else 'reverse'), flush=True)
