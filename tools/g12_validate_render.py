"""Known-answer check of the Python renderer: re-render the organiser's w058 surface and compare with the
organiser's published 24-layer render (README: layer NN = normal offset NN - 11.5 voxels).
Measures which convention (pixel offset, normal sign) reproduces it, and how exactly.
usage: g12_validate_render.py R0 C0 SIZE
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, time
import numpy as np, tifffile
sys.path.insert(0, _P.TOOLS)
import g12lib as L

R0, C0, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
D = f'{_P.DATA}/hf/ink-8um-pherc1447-surfaces/w058'
P, valid, meta = L.load_tifxyz(D + '/tifxyz')
N = L.vertex_normals(P)
pub = np.stack([tifffile.imread(f'{D}/layers/{i:02d}.tif')[R0:R0 + S, C0:C0 + S] for i in range(24)])
m = L.ChunkMirror()
rows, cols = np.arange(R0, R0 + S), np.arange(C0, C0 + S)
res = {}
for off in [0.5]:
    pts, nrm = L.surface_at_pixels(P, N, rows, cols, 0.25, 1.0, off)
    for flip in [False]:
        t = time.time()
        ours, v = L.render_pixels(m, pts, nrm, 24, 1.0, flip)
        k = v & (pub[12] > 0)
        a, b = ours[:, k].astype(np.float32), pub[:, k].astype(np.float32)
        per = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(24)]
        mad = float(np.abs(a - b).mean()); w1 = float((np.abs(a - b) <= 1).mean())
        res[f'off{off}_flip{flip}'] = dict(corr_mean=float(np.mean(per)), corr_min=float(np.min(per)), corr_l12=per[12],
                                          mad=mad, within1=w1, exact=float((a == b).mean()), n=int(k.sum()), secs=round(time.time() - t, 1))
        print(f'off={off} flip={flip}', json.dumps(res[f"off{off}_flip{flip}"]), flush=True)
best = max(res, key=lambda k: res[k]['corr_mean'])
print('BEST', best, 'fetched MB', round(m.fetched_bytes / 1e6))
json.dump(dict(tile=[R0, C0, S], results=res, best=best), open(f'{_P.OUT}/g1g2/render_validation.json', 'w'), indent=1)
