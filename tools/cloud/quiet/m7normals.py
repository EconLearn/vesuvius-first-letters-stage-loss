"""Label-free substitute for the organisers' lasagna fit inputs (nx, ny, grad_mag at OME level 2 = 4x) computed from
the organisers' PUBLISHED m7 surface prediction (binary sheet mask, th 0.2, nearest-downsampled pyramid).

Normals: 3D structure tensor of the m7 sheet mask at level 1 (2x, 17.3 um):
  gradient = Gaussian derivative (sigma_d, level-1 px) of the 0/1 mask;
  the 6 tensor products are mean-pooled 2x to level 2 and Gaussian-smoothed (sigma_i, level-2 px);
  normal = principal eigenvector (power iteration), hemisphere nz >= 0, uint8 = round(n*127+128), exactly the lasagna
  nx/ny encoding that spiral-fitting's _decode_uint8_normal reads (nz = sqrt(1-nx^2-ny^2)).
  Voxels with too little gradient energy (no sheet within ~2 sigma) get nx = ny = 0 (= "no data" for the fit).
grad_mag: local winding density from the m7 gap-distance field at level 1:
  D = EDT to the nearest sheet voxel, f = local sheet fraction, mD = local mean of D (Gaussian sigma_g);
  a periodic stack of pitch p and fill f has mean gap distance p (1-f)^2 / 4, so p = 4 mD / (1-f)^2 (level-1 px);
  density per full-res voxel = 1 / (2 p); uint8 = round(density * GM_SCALE), GM_SCALE calibrated once on PHerc0343's
  published grad_mag.

Only public, label-free inputs are read (anonymous S3). Nothing is written except where the caller says.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))    # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

# scratch space of the input builders: $QUIET_WORK/tmp (QUIET_WORK default $VC_WORK/quiet)
QUIET_WORK = os.environ.get('QUIET_WORK', os.path.join(_P.WORK, 'quiet'))
TMP = os.path.join(QUIET_WORK, 'tmp')

M7_TMPL = ('vesuvius-challenge-open-data/{s}/representations/predictions/surfaces/'
           '{v}-surface-20260413222639-surface-m7-L0-th0.2.zarr')
# Per-scroll CT volume id (VOLID; VOL = its timestamp, the first '-' separated part, which the m7 prediction paths use),
# array shape, voxel size in um and catalogue z-direction (True = z runs top to bottom, None = the catalogue has none).
# The public scrolls PHerc0343 and PHerc1447 are built in; every other scroll comes from the screening config
# ($SCREEN_CONFIG, default tools/cloud/screen_scrolls.tsv; columns: see that file). No path or size below assumes 8.64 um
# except these two built-in entries.
VOLID = {'PHerc0343': '20250521140437-8.640um-1.2m-116keV-masked.zarr', 'PHerc1447': '20250521151220-8.640um-1.2m-116keV-masked.zarr'}
VOL = {k: v.split('-')[0] for k, v in VOLID.items()}
SHAPE = {'PHerc0343': [17998, 8595, 8595], 'PHerc1447': [24297, 8343, 8343]}
VOXEL = {'PHerc0343': 8.64, 'PHerc1447': 8.64}
ZDIR = {'PHerc0343': False, 'PHerc1447': True}


def _load_screen_config(path=None):
    path = path or os.environ.get('SCREEN_CONFIG', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'screen_scrolls.tsv'))
    if not os.path.exists(path):
        return
    for line in open(path).read().splitlines():
        if not line.strip() or line.startswith('#') or line.startswith('dataset\t'):
            continue
        f = line.split('\t')
        scroll, vid = f[1], f[7]
        if vid == '-':
            continue
        VOLID.setdefault(scroll, vid)
        VOL.setdefault(scroll, vid.split('-')[0])
        VOXEL.setdefault(scroll, float(f[12]) if len(f) > 12 and f[12] not in ('', '-') else 8.64)
        SHAPE.setdefault(scroll, [int(f[8]), int(f[9]), int(f[10])])
        ZDIR.setdefault(scroll, {'true': True, 'false': False}.get(f[11]) if len(f) > 11 else None)


_load_screen_config()


def pool2(a):
    """Mean-pool by 2 along all three axes (shape must be even)."""
    z, y, x = a.shape
    return a.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))


def principal_eigvec(J, iters=12):
    """J: dict of tensor components (zz, zy, zx, yy, yx, xx) arrays. Returns (vz, vy, vx), lam1, trace."""
    zz, zy, zx, yy, yx, xx = (J[k] for k in ('zz', 'zy', 'zx', 'yy', 'yx', 'xx'))
    tr = zz + yy + xx
    # start from the row with the largest diagonal entry (robust start for sheet-like tensors)
    d = np.stack([zz, yy, xx])
    k = np.argmax(d, axis=0)
    vz = np.where(k == 0, zz, np.where(k == 1, zy, zx))
    vy = np.where(k == 0, zy, np.where(k == 1, yy, yx))
    vx = np.where(k == 0, zx, np.where(k == 1, yx, xx))
    for _ in range(iters):
        nz_ = zz * vz + zy * vy + zx * vx
        ny_ = zy * vz + yy * vy + yx * vx
        nx_ = zx * vz + yx * vy + xx * vx
        n = np.sqrt(nz_ * nz_ + ny_ * ny_ + nx_ * nx_) + 1e-30
        vz, vy, vx = nz_ / n, ny_ / n, nx_ / n
    lam1 = (zz * vz + zy * vy + zx * vx) * vz + (zy * vz + yy * vy + yx * vx) * vy + (zx * vz + yx * vy + xx * vx) * vx
    return (vz, vy, vx), lam1, tr


def compute_block(m7_l1, sigma_d=1.0, sigma_i=2.5, energy_min=0.02, want_density=True, ray_half=16, fast_grad=True):
    """m7_l1: uint8/bool level-1 block (z, y, x) with even shape INCLUDING halo (halo >= max(ray_half, 3*2*sigma_i)
    level-1 px). Returns level-2 arrays (same halo, halved): normal (vz, vy, vx) float32, valid bool, coherence,
    density per full-res voxel (ray-crossing estimate, or None)."""
    S = (np.asarray(m7_l1) > 0)
    f = S.astype(np.float32)
    if fast_grad:
        # Gaussian smoothing once, then central differences (= Gaussian derivative up to the difference stencil)
        g = ndimage.gaussian_filter(f, sigma_d, mode='nearest')
        gz = np.empty_like(g); gy = np.empty_like(g); gx = np.empty_like(g)
        gz[1:-1] = (g[2:] - g[:-2]) * 0.5; gz[0] = gz[1]; gz[-1] = gz[-2]
        gy[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5; gy[:, 0] = gy[:, 1]; gy[:, -1] = gy[:, -2]
        gx[:, :, 1:-1] = (g[:, :, 2:] - g[:, :, :-2]) * 0.5; gx[:, :, 0] = gx[:, :, 1]; gx[:, :, -1] = gx[:, :, -2]
        del g
    else:
        gz = ndimage.gaussian_filter(f, sigma_d, order=(1, 0, 0), mode='nearest')
        gy = ndimage.gaussian_filter(f, sigma_d, order=(0, 1, 0), mode='nearest')
        gx = ndimage.gaussian_filter(f, sigma_d, order=(0, 0, 1), mode='nearest')
    J = {}
    for name, a, b in (('zz', gz, gz), ('zy', gz, gy), ('zx', gz, gx), ('yy', gy, gy), ('yx', gy, gx), ('xx', gx, gx)):
        J[name] = ndimage.gaussian_filter(pool2(a * b), sigma_i, mode='nearest').astype(np.float32)
    del gz, gy, gx
    (vz, vy, vx), lam1, tr = principal_eigvec(J)
    coh = lam1 / (tr + 1e-30)  # 1 = perfectly planar, 1/3 = isotropic
    # energy normaliser: a sharp 0/1 edge blurred by sigma_d has peak |grad|^2 ~ 1/(2 pi sigma_d^2)
    energy = tr * (2 * np.pi * sigma_d ** 2)
    valid = energy > energy_min
    dens = None
    if want_density:
        dens = ray_density(S, (vz, vy, vx), valid, ray_half)
    return (vz.astype(np.float32), vy.astype(np.float32), vx.astype(np.float32)), valid, coh.astype(np.float32), dens


def ray_density(S, v, valid, T=16, step=0.5):
    """Winding density per FULL-RES voxel at each level-2 voxel: count sheet entries (0->1 transitions of the binary
    level-1 m7 mask) along the local normal over [-T, T] level-1 px, divided by the ray length in full-res voxels."""
    vz, vy, vx = v
    Z, Y, X = vz.shape
    zi, yi, xi = np.nonzero(valid)
    out = np.zeros((Z, Y, X), np.float32)
    if not len(zi):
        return out
    ts = np.arange(-T, T + step / 2, step, dtype=np.float32)
    cz = zi * 2 + 0.5; cy = yi * 2 + 0.5; cx = xi * 2 + 0.5   # level-2 voxel centre in level-1 coordinates
    nz_, ny_, nx_ = vz[zi, yi, xi], vy[zi, yi, xi], vx[zi, yi, xi]
    cnt = np.zeros(len(zi), np.int32)
    prev = None
    Sb = S.astype(np.uint8)
    for t in ts:
        pz = np.clip(np.rint(cz + t * nz_), 0, S.shape[0] - 1).astype(np.int32)
        py = np.clip(np.rint(cy + t * ny_), 0, S.shape[1] - 1).astype(np.int32)
        px = np.clip(np.rint(cx + t * nx_), 0, S.shape[2] - 1).astype(np.int32)
        cur = Sb[pz, py, px]
        if prev is not None:
            cnt += (cur > prev)
        prev = cur
    out[zi, yi, xi] = cnt / (2.0 * T * 2.0)
    return out


def encode_u8(vz, vy, vx, valid):
    flip = np.where(vz < 0, -1.0, 1.0)
    nx = np.clip(np.round(vx * flip * 127.0 + 128.0), 0, 255).astype(np.uint8)
    ny = np.clip(np.round(vy * flip * 127.0 + 128.0), 0, 255).astype(np.uint8)
    nx[~valid] = 0
    ny[~valid] = 0
    # a valid voxel must never decode as "no data"
    both0 = valid & (nx == 0) & (ny == 0)
    nx[both0] = 1
    return nx, ny


def decode_u8(nx_u8, ny_u8):
    """Exactly spiral-fitting losses._decode_uint8_normal: returns (nz, ny, nx) unit, valid."""
    valid = (nx_u8 != 0) | (ny_u8 != 0)
    nx = (nx_u8.astype(np.float32) - 128.0) / 127.0
    ny = (ny_u8.astype(np.float32) - 128.0) / 127.0
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nz / n, ny / n, nx / n, valid


def angle_deg(a, b):
    """Sign-free angle between unit vector fields a=(z,y,x), b=(z,y,x)."""
    c = np.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


class ChunkMirror:
    """Process-safe local mirror of a public zarr v2 array on S3 (anonymous). Chunks are fetched on demand, written
    atomically (tmp + rename), and decoded with numcodecs. Missing chunks (fill value) are remembered.
    read_box(z0, z1, y0, y1, x0, x1) -> ndarray (out-of-range parts filled with fill_value)."""

    def __init__(self, s3_array_path, local_dir, workers=16):
        import json, os, s3fs, numcodecs
        self.s3p = s3_array_path.rstrip('/')
        self.loc = local_dir.rstrip('/')
        os.makedirs(self.loc, exist_ok=True)
        self.fs = s3fs.S3FileSystem(anon=True)
        za = os.path.join(self.loc, '.zarray')
        if not os.path.exists(za):
            b = self.fs.cat_file(self.s3p + '/.zarray')
            tmp = za + f'.{os.getpid()}.part'
            open(tmp, 'wb').write(b); os.replace(tmp, za)
        meta = json.load(open(za))
        self.shape = tuple(meta['shape']); self.chunks = tuple(meta['chunks'])
        self.dtype = np.dtype(meta['dtype']); self.fill = meta['fill_value'] or 0
        self.sep = meta.get('dimension_separator', '.')
        self.codec = numcodecs.get_codec(meta['compressor'])
        self.workers = workers

    def _key(self, c):
        return self.sep.join(str(i) for i in c)

    def _local(self, c):
        import os
        return os.path.join(self.loc, *[str(i) for i in c]) if self.sep == '/' else os.path.join(self.loc, self._key(c))

    def _fetch(self, c):
        import os, time
        p = self._local(c)
        if os.path.exists(p) or os.path.exists(p + '.absent'):
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for a in range(6):
            try:
                b = self.fs.cat_file(f'{self.s3p}/{self._key(c)}')
                break
            except FileNotFoundError:
                open(p + '.absent', 'w').close(); return
            except Exception:
                time.sleep(2 * (a + 1))
        else:
            raise RuntimeError('fetch failed ' + self._key(c))
        tmp = f'{p}.{os.getpid()}.part'
        open(tmp, 'wb').write(b); os.replace(tmp, p)

    def _chunk(self, c):
        import os
        p = self._local(c)
        if os.path.exists(p + '.absent'):
            return None
        raw = self.codec.decode(open(p, 'rb').read())
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks)

    def read_box(self, z0, z1, y0, y1, x0, x1):
        from concurrent.futures import ThreadPoolExecutor
        lo = (z0, y0, x0); hi = (z1, y1, x1)
        out = np.full((z1 - z0, y1 - y0, x1 - x0), self.fill, dtype=self.dtype)
        clo = [max(lo[i], 0) for i in range(3)]; chi = [min(hi[i], self.shape[i]) for i in range(3)]
        if any(clo[i] >= chi[i] for i in range(3)):
            return out
        rng = [range(clo[i] // self.chunks[i], (chi[i] - 1) // self.chunks[i] + 1) for i in range(3)]
        keys = [(a, b, c) for a in rng[0] for b in rng[1] for c in rng[2]]
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self._fetch, keys))
        for k in keys:
            ch = self._chunk(k)
            if ch is None:
                continue
            s = [k[i] * self.chunks[i] for i in range(3)]
            a0 = [max(clo[i], s[i]) for i in range(3)]; a1 = [min(chi[i], s[i] + self.chunks[i]) for i in range(3)]
            out[a0[0] - lo[0]:a1[0] - lo[0], a0[1] - lo[1]:a1[1] - lo[1], a0[2] - lo[2]:a1[2] - lo[2]] = \
                ch[a0[0] - s[0]:a1[0] - s[0], a0[1] - s[1]:a1[1] - s[1], a0[2] - s[2]:a1[2] - s[2]]
        return out
