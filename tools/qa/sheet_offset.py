"""sheet_offset.py MESH_SPECS K OUT_JSON
How far does a mesh sit from the predicted papyrus sheet? Sampling version of the measurement inside Nieuwlaar's
snap_meshes.py (github.com/Nieuwlaar/pherc343-first-letters, MIT): for each vertex, sample the published m7 surface prediction (binary
sheet mask, level 0) along the mesh normal at -15..+15 voxels and take the centre of the nearest sheet run.
K random m7 chunks crossed by the mesh are streamed from public S3 into memory (nothing is written to disk).
MESH_SPECS: comma list of name=SCROLL:path_to_tifxyz[@shift_voxels]. The @shift arm is the built-in sabotage:
a mesh moved off the sheet on purpose must score worse. Validation and results: FINDINGS section 8."""
import sys, json, numpy as np, tifffile, s3fs, numcodecs
from pathlib import Path
KOFF = 15; OFFS = np.arange(-KOFF, KOFF + 1)
fs = s3fs.S3FileSystem(anon=True)

def m7_level0(scroll):
    base = f'vesuvius-challenge-open-data/{scroll}/representations/predictions/surfaces/'
    names = sorted(p.rstrip('/').split('/')[-1] for p in fs.ls(base))
    m7 = [n for n in names if n.endswith('-surface-m7-L0-th0.2.zarr')]
    if not m7:
        raise SystemExit(f'{scroll}: no m7 L0 th0.2 surface prediction on the bucket')
    return base + m7[-1] + '/0'


class Arr:
    def __init__(self, scroll):
        self.p = m7_level0(scroll)
        m = json.loads(fs.cat_file(self.p + '/.zarray'))
        self.shape, self.ch = tuple(m['shape']), tuple(m['chunks'])
        self.sep = m.get('dimension_separator', '.'); self.codec = numcodecs.get_codec(m['compressor']) if m['compressor'] else None
        self.dtype = np.dtype(m['dtype'])
    def chunk(self, c):
        try:
            b = fs.cat_file(f"{self.p}/{self.sep.join(map(str, c))}")
        except FileNotFoundError:
            return np.zeros(self.ch, self.dtype)
        raw = self.codec.decode(b) if self.codec else b
        return np.frombuffer(raw, self.dtype).reshape(self.ch)

def normals(X, Y, Zs, valid):
    P = np.stack([Zs, Y, X], -1).astype(np.float64); P[~valid] = np.nan
    tu = np.gradient(P, axis=1); tv = np.gradient(P, axis=0)
    n = np.cross(tu, tv); nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(nn > 1e-6, nn, np.nan)
    return P, n, np.isfinite(n).all(-1) & valid

def nearest_run(prof):
    if not prof.any(): return np.nan
    pad = np.concatenate([[0], prof.astype(np.int8), [0]]); dif = np.diff(pad)
    st, en = np.nonzero(dif == 1)[0], np.nonzero(dif == -1)[0]
    cen = (st + en - 1) / 2.0 - KOFF
    return float(cen[np.argmin(np.abs(cen))])

def run(name, scroll, path, shift, K, seed=0):
    A = Arr(scroll)
    d = Path(path)
    X, Y, Zs = (tifffile.imread(d / f'{a}.tif').astype(np.float64) for a in 'xyz')
    valid = (X > -0.5) & (Y > -0.5) & (Zs > -0.5) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Zs) & (Zs > 0)
    P, N, ok = normals(X, Y, Zs, valid)
    P = P + shift * np.nan_to_num(N)
    vy, vx = np.nonzero(ok); pts = P[vy, vx]; nr = N[vy, vx]
    ch = np.array(A.ch)
    cid = (pts // ch).astype(np.int64)
    margin = KOFF + 1
    inner = np.all((pts - cid * ch >= margin) & (pts - cid * ch < ch - margin), axis=1)
    keys, inv = np.unique(cid[inner], axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(keys), size=min(K, len(keys)), replace=False)
    idx_inner = np.nonzero(inner)[0]
    offs, nonempty = [], 0
    for k in pick:
        blk = A.chunk(tuple(keys[k]))
        if not blk.any(): continue
        nonempty += 1
        sel = idx_inner[inv.ravel() == k]
        lo = keys[k] * ch
        S = pts[sel][:, None, :] + OFFS[None, :, None] * nr[sel][:, None, :] - lo
        Si = np.rint(S).astype(np.int64)
        v = blk[Si[..., 0], Si[..., 1], Si[..., 2]] > 0
        offs += [nearest_run(p) for p in v]
    o = np.array(offs, float); has = np.isfinite(o)
    ao = np.abs(o[has])
    return dict(name=name, scroll=scroll, shift=shift, chunks=int(nonempty), vertices=int(len(o)),
                frac_with_sheet=float(has.mean()) if len(o) else None,
                frac_on_le2=float((ao <= 2).mean()) if has.any() else None,
                frac_off_ge4=float((ao >= 4).mean()) if has.any() else None,
                median_abs_off=float(np.median(ao)) if has.any() else None,
                p90_abs_off=float(np.percentile(ao, 90)) if has.any() else None,
                signed_median=float(np.median(o[has])) if has.any() else None)

if __name__ == '__main__':
    specs, K, out = sys.argv[1].split(','), int(sys.argv[2]), sys.argv[3]
    res = []
    for sp in specs:
        name, rest = sp.split('=', 1); scroll, rest = rest.split(':', 1)
        path, _, sh = rest.partition('@')
        r = run(name, scroll, path, float(sh or 0), K); res.append(r); print(json.dumps(r), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))
