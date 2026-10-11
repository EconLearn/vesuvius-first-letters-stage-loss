"""G1: automatic umbilicus for PHerc1447 (no published umbilicus exists for this scroll).

Port of VC3D core/src/normalgridtools.cpp::align_and_extract_umbilicus (the function vc_gen_umbilicus runs):
  - 10,000 sampled sheet points with in-plane sheet normals,
  - RANSAC: 1,000 uniform candidates, score = sum cos^2(angle(point - c, normal)),
  - hill climb from step 1024 down to 1 on score = sum cos^2 / max(100, dist).
Inputs (label-free, published): m7 surface prediction (sheet mask) at level 2 -> skeleton = sheet "paths";
normals = published lasagna nx/ny at level 2 (decoded (v-128)/127, in-plane direction only; cos^2 is sign-free).
Coordinates: full-res voxels (level-2 index * 4 + 1.5, pixel centre).
NO organiser surface or label is read here.
usage: g12_umbilicus.py OUT.json Z0 Z1 STEP [seeds...]
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, time
import numpy as np, s3fs, zarr
from skimage.morphology import skeletonize

OUT, Z0, Z1, STEP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5:]] or [1, 2, 3]
fs = s3fs.S3FileSystem(anon=True)
M7 = 'vesuvius-challenge-open-data/PHerc1447/representations/predictions/surfaces/20250521151220-surface-20260413222639-surface-m7-L0-th0.2.zarr/2'
m7 = zarr.open_array(store=zarr.storage.FsspecStore(fs, path=M7, read_only=True), mode='r') if hasattr(zarr, 'storage') and hasattr(zarr.storage, 'FsspecStore') else zarr.open(fs.get_mapper(M7), mode='r')
LAS = f'{_P.DATA}/spiral/PHerc1447/lasagna_inputs'
nxa = zarr.open(f'{LAS}/PHerc1447_nx.ome.zarr/2', mode='r')
nya = zarr.open(f'{LAS}/PHerc1447_ny.ome.zarr/2', mode='r')


def estimate(pts, nrm, extent, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pts), 10000)
    P, N = pts[idx].astype(np.float64), nrm[idx].astype(np.float64)
    cand = np.stack([rng.uniform(0, extent[0], 1000), rng.uniform(0, extent[1], 1000)], 1)
    best, bs = None, -1
    for c in cand:
        d = P - c; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        s = ((d * N).sum(1) ** 2).sum()
        if s > bs:
            bs, best = s, c

    def score(c):
        d = P - c; dist = np.linalg.norm(d, axis=1); ok = dist > 1e-6
        d = d[ok] / dist[ok, None]
        return (((d * N[ok]).sum(1) ** 2) / np.maximum(100.0, dist[ok])).sum()
    b = best.copy(); sb = score(b); step = 1024.0
    while step >= 1.0:
        moved = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                c = b + np.array([dx * step, dy * step]); s = score(c)
                if s > sb:
                    sb, b, moved = s, c, True
        if not moved:
            step /= 2
    return b, sb


res = []
for z in range(Z0, Z1 + 1, STEP):
    t = time.time(); zl = z // 4
    sheet = np.asarray(m7[zl]) > 0
    nx = np.asarray(nxa[zl]).astype(np.float32); ny = np.asarray(nya[zl]).astype(np.float32)
    valid = (nx != 0) | (ny != 0)
    sk = skeletonize(sheet) & valid
    vx = (nx - 128) / 127; vy = (ny - 128) / 127
    nn = np.sqrt(vx ** 2 + vy ** 2)
    yy, xx = np.nonzero(sk & (nn > 0.3))           # in-plane component must carry a direction
    N = np.stack([vy[yy, xx], vx[yy, xx]], 1) / nn[yy, xx, None]   # (y, x) order
    P = np.stack([yy * 4 + 1.5, xx * 4 + 1.5], 1)                 # full-res (y, x)
    ests = []
    for sd in SEEDS:
        c, s = estimate(P, N, (sheet.shape[0] * 4, sheet.shape[1] * 4), sd)
        ests.append([float(c[0]), float(c[1]), float(s)])
    E = np.array(ests)
    rec = dict(z=z, n_sheet_px=int(sheet.sum()), n_skel=int(len(P)), seeds=SEEDS, yx_per_seed=E[:, :2].tolist(),
               yx_median=np.median(E[:, :2], 0).tolist(), spread_vox=float(np.ptp(E[:, :2], 0).max()), secs=round(time.time() - t, 1))
    print(json.dumps(rec), flush=True)
    res.append(rec)
cps = [dict(x=round(r['yx_median'][1], 1), y=round(r['yx_median'][0], 1), z=r['z'], score=100) for r in res]
json.dump(dict(control_points=cps, provenance=dict(method='port of VC3D align_and_extract_umbilicus (RANSAC 1000 x 10000 + weighted hill climb)',
                                                    inputs='PHerc1447 m7 L2 skeleton + lasagna nx/ny L2 (published, label-free)',
                                                    per_slice=res)), open(OUT, 'w'), indent=1)
print('wrote', OUT)
