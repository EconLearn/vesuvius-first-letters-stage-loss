"""G1/G2 shared code: tifxyz IO, on-demand CT chunk mirror, vc_render_tifxyz-equivalent renderer.

Renderer semantics follow volume-cartographer apps/src/vc_render_tifxyz.cpp + core QuadSurface::gen (linear mode):
  - positions: bilinear interpolation of the tifxyz grid at grid coord = pixel * (grid_scale / render_scale) + PIX_OFF
  - normals: per-vertex grid_normal_int = (P[r,c+1]-P[r,c-1]) x (P[r+1,c]-P[r-1,c]), NaN on border/invalid
    vertices, looked up nearest-neighbour; --flip-normals negates; then normalised
  - layers: offsets (i - (n-1)/2) * step along the unit normal, volume sampled trilinearly (voxel i at coord i)
The pixel-centre convention (grid = (pixel + 0.5) * scale) and uint8 truncation are measured against the organisers' own 1447 renders (g12_validate_render.py).
"""
from __future__ import annotations
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage

VOL_ID = '20250521151220-8.640um-1.2m-116keV-masked.zarr'
S3_VOL = f'vesuvius-challenge-open-data/PHerc1447/volumes/{VOL_ID}/0'
MIRROR = Path(f'{_P.DATA}/s3/PHerc1447/volumes/{VOL_ID}/0')
SHAPE = (24297, 8343, 8343); CH = 128


def load_tifxyz(d):
    d = Path(d)
    x = tifffile.imread(d / 'x.tif').astype(np.float32); y = tifffile.imread(d / 'y.tif').astype(np.float32)
    z = tifffile.imread(d / 'z.tif').astype(np.float32)
    meta = json.loads((d / 'meta.json').read_text())
    P = np.stack([x, y, z], -1)
    valid = (x != -1) & np.isfinite(x) & ~((x == 0) & (y == 0) & (z == 0))
    P[~valid] = np.nan
    return P, valid, meta


def vertex_normals(P):
    """grid_normal_int: (P[r,c+1]-P[r,c-1]) x (P[r+1,c]-P[r-1,c]); NaN on borders / invalid neighbours. XYZ order."""
    N = np.full(P.shape, np.nan, np.float32)
    xv = P[1:-1, 2:] - P[1:-1, :-2]
    yv = P[2:, 1:-1] - P[:-2, 1:-1]
    N[1:-1, 1:-1] = np.cross(xv, yv)
    return N


class ChunkMirror:
    """Local mirror of the L0 CT zarr (128^3 uint8, uncompressed). Fetches missing chunks from public S3."""
    def __init__(self, workers=32):
        import s3fs
        self.fs = s3fs.S3FileSystem(anon=True)
        self.absent_f = MIRROR.parent / '.absent_L0.txt'
        self.absent = set(self.absent_f.read_text().split()) if self.absent_f.exists() else set()
        self.lock = threading.Lock(); self.workers = workers
        MIRROR.mkdir(parents=True, exist_ok=True)
        za = MIRROR / '.zarray'
        if not za.exists():
            za.write_bytes(self.fs.cat_file(S3_VOL + '/.zarray'))
        self.fetched_bytes = 0

    def path(self, k):
        return MIRROR / f'{k[0]}/{k[1]}/{k[2]}'

    def _get(self, k):
        key = f'{k[0]}/{k[1]}/{k[2]}'
        p = self.path(k)
        if p.exists() or key in self.absent:
            return
        for a in range(6):
            try:
                b = self.fs.cat_file(f'{S3_VOL}/{key}')
                break
            except FileNotFoundError:
                with self.lock:
                    self.absent.add(key)
                    with open(self.absent_f, 'a') as f:
                        f.write(key + '\n')
                return
            except Exception:
                time.sleep(2 * (a + 1))
        else:
            raise RuntimeError('fetch failed ' + key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if len(b) != CH ** 3:
            raise RuntimeError(f'bad chunk size {key}: {len(b)}')
        tmp = p.parent / f'{p.name}.{os.getpid()}.{threading.get_ident()}.part'
        tmp.write_bytes(b); os.replace(tmp, p)
        with self.lock:
            self.fetched_bytes += len(b)

    def ensure(self, keys):
        keys = [k for k in keys if not self.path(k).exists() and f'{k[0]}/{k[1]}/{k[2]}' not in self.absent]
        if keys:
            with ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(self._get, keys))
        return len(keys)

    def read_box(self, z0, z1, y0, y1, x0, x1):
        """Read [z0:z1, y0:y1, x0:x1] (clipped to the volume; outside = 0), fetching chunks as needed."""
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        cz = range(max(0, z0) // CH, (min(SHAPE[0], z1) - 1) // CH + 1)
        cy = range(max(0, y0) // CH, (min(SHAPE[1], y1) - 1) // CH + 1)
        cx = range(max(0, x0) // CH, (min(SHAPE[2], x1) - 1) // CH + 1)
        keys = [(a, b, c) for a in cz for b in cy for c in cx]
        self.ensure(keys)
        for (a, b, c) in keys:
            p = self.path((a, b, c))
            if not p.exists():
                continue
            ch = np.frombuffer(p.read_bytes(), np.uint8).reshape(CH, CH, CH)
            za, ya, xa = a * CH, b * CH, c * CH
            sz0, sz1 = max(z0, za), min(z1, za + CH); sy0, sy1 = max(y0, ya), min(y1, ya + CH); sx0, sx1 = max(x0, xa), min(x1, xa + CH)
            if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
                continue
            out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = ch[sz0 - za:sz1 - za, sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa]
        return out


def surface_at_pixels(P, N, rows, cols, grid_scale, render_scale=1.0, pix_off=0.5):
    """3D points (bilinear) and unit normals (nearest vertex) for render pixels rows x cols (1-D index arrays).
    Returns pts (len(rows), len(cols), 3) XYZ and nrm same shape (NaN where invalid)."""
    s = grid_scale / render_scale
    # pixel-centre convention (measured vs organiser render): grid = (pixel + pix_off) * s, pix_off = 0.5
    gr = (rows.astype(np.float64) + pix_off) * s; gc = (cols.astype(np.float64) + pix_off) * s
    GR, GC = np.meshgrid(gr, gc, indexing='ij')
    r0 = np.floor(GR).astype(np.int64); c0 = np.floor(GC).astype(np.int64)
    fr = (GR - r0)[..., None]; fc = (GC - c0)[..., None]
    H, W = P.shape[:2]
    ok = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < H) & (c0 + 1 < W)
    r0c = np.clip(r0, 0, H - 2); c0c = np.clip(c0, 0, W - 2)
    p00 = P[r0c, c0c]; p01 = P[r0c, c0c + 1]; p10 = P[r0c + 1, c0c]; p11 = P[r0c + 1, c0c + 1]
    pts = (p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc).astype(np.float32)
    rn = np.clip(np.rint(GR).astype(np.int64), 0, H - 1); cn = np.clip(np.rint(GC).astype(np.int64), 0, W - 1)
    nrm = N[rn, cn].astype(np.float32)
    nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    pts[~ok] = np.nan; nrm[~ok] = np.nan
    return pts, nrm


def _render_tile(mirror, p, n, v, offs):
    n_slices = len(offs)
    S = p[None] + offs[:, None, None, None] * n[None]          # (k, h, w, 3) XYZ
    Sv = S[:, v]                                                # (k, m, 3)
    lo = np.floor(np.nanmin(Sv.reshape(-1, 3), 0)).astype(int) - 1
    hi = np.ceil(np.nanmax(Sv.reshape(-1, 3), 0)).astype(int) + 2
    if np.any(hi - lo > 1500):        # degenerate tile (folded / exploding mesh): skip rather than read a huge box
        return None
    box = mirror.read_box(lo[2], hi[2], lo[1], hi[1], lo[0], hi[0])
    coords = np.stack([Sv[..., 2] - lo[2], Sv[..., 1] - lo[1], Sv[..., 0] - lo[0]], 0).reshape(3, -1)
    vals = ndimage.map_coordinates(box, coords, order=1, mode='constant', cval=0, prefilter=False)
    blk = np.zeros((n_slices,) + v.shape, np.float32)
    blk[:, v] = vals.reshape(n_slices, -1)
    return np.clip(np.floor(blk + 1e-4), 0, 255).astype(np.uint8)  # C++ truncation (measured)


_W = {}


def _worker_tile(args):
    i0, j0, p, n, v, offs = args
    if 'm' not in _W:
        _W['m'] = ChunkMirror(workers=8)
    return i0, j0, _render_tile(_W['m'], p, n, v, offs)


def render_pixels(mirror, pts, nrm, n_slices, step=1.0, flip=False, tile=192, extra_offset=0.0, procs=0):
    """Render layers for a (H, W) block of surface points. Returns (n_slices, H, W) uint8, valid mask (H, W).
    procs > 0: tiles rendered in a fork process pool (each worker has its own chunk mirror handle)."""
    H, W = pts.shape[:2]
    offs = (np.arange(n_slices) - 0.5 * (n_slices - 1)) * step + extra_offset
    d = -nrm if flip else nrm
    out = np.zeros((n_slices, H, W), np.uint8)
    valid = np.isfinite(pts).all(-1) & np.isfinite(d).all(-1)
    jobs = []
    for i0 in range(0, H, tile):
        for j0 in range(0, W, tile):
            v = valid[i0:i0 + tile, j0:j0 + tile]
            if v.any():
                jobs.append((i0, j0, pts[i0:i0 + tile, j0:j0 + tile], d[i0:i0 + tile, j0:j0 + tile], v, offs))
    if procs and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(procs) as pool:
            for i0, j0, blk in pool.imap_unordered(_worker_tile, jobs, chunksize=2):
                if blk is not None:
                    out[:, i0:i0 + blk.shape[1], j0:j0 + blk.shape[2]] = blk
    else:
        for (i0, j0, p, n, v, o) in jobs:
            blk = _render_tile(mirror, p, n, v, o)
            if blk is not None:
                out[:, i0:i0 + blk.shape[1], j0:j0 + blk.shape[2]] = blk
    return out, valid
