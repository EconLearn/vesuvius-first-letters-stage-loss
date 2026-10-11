"""In-plane orientation of an automatic render relative to the organiser render, at matched points.
For organiser pixel q: unit 3D directions of its +col (u) and +row (v) axes; for the matched automatic pixel: its +col (a)
and +row (b) axes. Express a in the organiser (u, v) basis -> rotation angle of the auto column axis; handedness =
sign of det[[a.u, a.v], [b.u, b.v]] (negative = mirrored text); scale = |dP/dpixel| ratio.
usage: g12_orient.py TAG"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
tag = sys.argv[1]
meta = json.loads((G.ARMS / f'auto_{tag}.json').read_text())
P, A, B = [], [], []
for c in meta['comps']:
    pos = np.load(G.ARMS / f'{c["name"]}_pts.npy')
    a = np.gradient(pos, axis=1); b = np.gradient(pos, axis=0)
    ok = np.isfinite(pos).all(-1) & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    P.append(pos[ok]); A.append(a[ok]); B.append(b[ok])
P = np.concatenate(P); A = np.concatenate(A); B = np.concatenate(B)
T = cKDTree(P, balanced_tree=False, compact_nodes=False)
out = {}
for w in G.WINDS:
    g = np.load(G.W / 'org' / w / 'geo.npz'); band = g['band']
    pts = g['pts']; U = np.gradient(pts, axis=1); V = np.gradient(pts, axis=0)
    Q = pts[band]; u = U[band]; v = V[band]
    d, j = T.query(Q, distance_upper_bound=14, workers=4); hit = np.isfinite(d)
    j = j[hit]; u = u[hit]; v = v[hit]
    un = u / np.linalg.norm(u, axis=1, keepdims=True); vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    a = A[j]; b = B[j]
    au, av, bu, bv = (a * un).sum(1), (a * vn).sum(1), (b * un).sum(1), (b * vn).sum(1)
    ang = np.degrees(np.arctan2(av, au))                     # rotation of the auto column axis in organiser (u,v)
    det = au * bv - av * bu
    scale_a = np.linalg.norm(a, axis=1) / np.linalg.norm(u, axis=1)
    scale_b = np.linalg.norm(b, axis=1) / np.linalg.norm(v, axis=1)
    hist, edges = np.histogram(ang, bins=12, range=(-180, 180))
    out[w] = dict(n=int(hit.sum()), mirrored_frac=float((det < 0).mean()),
                  angle_pct=[float(x) for x in np.percentile(ang, [5, 25, 50, 75, 95])],
                  angle_hist_30deg=hist.tolist(), scale_col_median=float(np.median(scale_a)), scale_row_median=float(np.median(scale_b)))
    print(tag, w, json.dumps(out[w]), flush=True)
json.dump(out, open(f'{_P.OUT}/g1g2/orient_{tag}.json', 'w'), indent=1)
