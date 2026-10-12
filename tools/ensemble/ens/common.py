"""Ensemble shared conventions (Nieuwlaar PHerc 343 First Letters recipe, ported to our pipeline).

Depth convention (his README + mesh/render_66_layers.sh, verified against our renderer):
  * an N-layer render samples layer k at offset (k - (N-1)/2) * step along the unit surface normal
    (vc_render_tifxyz; our g12lib port does the same). For N = 66 the surface sits between layers 32 and 33.
  * depth d = k - N//2  (d = k - 33 for N = 66), so layer k = 33 + d is at offset d + 0.5 voxels.
  * the normal is oriented so that +d points AWAY from the scroll axis (render66 checks this against the umbilicus
    and flips when needed; the flip decision is stored in the zarr attrs).
  * our 28-layer renders sample offset j - 13.5, i.e. 28-layer j == 66-layer k = j + 19.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Paths come from tools/vcpaths.py ($VC_ROOT, $VC_DATA, $VC_WORK, $VC_CKPT, $VILLA_DIR ...).
_TOOLS = Path(__file__).resolve().parents[2]            # <repo>/tools: vcpaths.py and g12lib.py live there
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

LANE = Path(__file__).resolve().parents[1]              # tools/ensemble (this package's parent directory)
WORK = Path(_P.WORK) / 'ensemble'                       # default work dir for ensemble runs: $VC_WORK/ensemble
PY = sys.executable
G12 = str(_TOOLS)                                       # directory holding g12lib.py
VILLA_SRC = os.path.join(_P.VILLA, 'vesuvius', 'src')   # $VILLA_DIR/vesuvius/src
CKPT = Path(_P.CKPT)
V8IN = Path(_P.V8IN_DIR)                                # ink8um package + v8-in weights ($V8IN_DIR)
# Released weights and the winning submission's repo live under $VC_EXT (default $VC_ROOT/ext).
EXT_ROOT = Path(os.environ.get('VC_EXT', os.path.join(_P.ROOT, 'ext')))
N66 = 66

VOLUMES = {
    'PHerc1447': dict(vol_id='20250521151220-8.640um-1.2m-116keV-masked.zarr', shape=(24297, 8343, 8343), um=8.64,
                      umbilicus=f'{_P.DATA}/spiral/PHerc1447/umbilicus.json'),
    'PHerc0343': dict(vol_id='20250521140437-8.640um-1.2m-116keV-masked.zarr', shape=(17998, 8595, 8595), um=8.64,
                      umbilicus=f'{_P.DATA}/spiral/PHerc0343/umbilicus.json'),
}


def layer_of_depth(d: int, n: int = N66) -> int:
    return n // 2 + int(d)


def depth_of_layer(k: int, n: int = N66) -> int:
    return int(k) - n // 2


def offset_of_layer(k: float, n: int = N66, step: float = 1.0) -> float:
    return (k - 0.5 * (n - 1)) * step


def parse_depths(spec) -> list[int]:
    """'-6..6:2' or '-2,0,2' or ['-15..15'] -> sorted ints, centre-out order is applied by callers."""
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out += parse_depths(s)
        return sorted(set(out))
    s = str(spec).strip()
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '..' in part:
            rng, _, st = part.partition(':')
            a, b = rng.split('..')
            out += list(range(int(a), int(b) + 1, int(st) if st else 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def centre_out(depths):
    """Visit 0, -1, +1, -2, +2 ... (his run_depth_sweep order) so a partial run already covers central depths."""
    return sorted(depths, key=lambda d: (abs(d), d > 0))


def use_volume(scroll: str, allow_fetch: bool | None = None):
    """Point the validated g12lib renderer at a scroll's CT mirror. Fetching missing CT chunks from S3 is OFF unless
    allow_fetch (or env ENS_ALLOW_FETCH=1): a render that needs an absent chunk then fails with the chunk count, so
    nothing is ever downloaded without the explicit switch."""
    sys.path.insert(0, G12)
    import g12lib as L
    v = VOLUMES[scroll]
    L.VOL_ID = v['vol_id']
    L.S3_VOL = f"vesuvius-challenge-open-data/{scroll}/volumes/{v['vol_id']}/0"
    L.MIRROR = Path(f"{_P.DATA}/s3/{scroll}/volumes/{v['vol_id']}/0")
    L.SHAPE = tuple(v['shape'])
    za = json.loads((L.MIRROR / '.zarray').read_text())
    if tuple(za['shape']) != tuple(L.SHAPE) or tuple(za['chunks']) != (128, 128, 128) or za['compressor'] is not None:
        raise SystemExit(f'{scroll}: CT mirror metadata mismatch {za}')
    if allow_fetch is None:
        allow_fetch = os.environ.get('ENS_ALLOW_FETCH', '0') == '1'
    if not allow_fetch:
        _orig = L.ChunkMirror._get

        def _get_offline(self, k):
            key = f'{k[0]}/{k[1]}/{k[2]}'
            if self.path(k).exists() or key in self.absent:
                return
            raise RuntimeError(f'CT chunk {key} of {scroll} is not mirrored and fetching is disabled '
                               f'(set ENS_ALLOW_FETCH=1 only after approving the CT download)')
        L.ChunkMirror._get = _get_offline
        L.ChunkMirror._get_online = _orig
    L._W.clear()
    return L


# the winning submission's MIT repo (data read only; its scripts are never run); override with $PHERC343_REPO
HIS_REPO = Path(os.environ.get('PHERC343_REPO', str(EXT_ROOT / 'pherc343-first-letters')))


def rel_file(rel, name):
    """Find a file/dir of his release in REL, REL/outputs or REL/inputs (repo layout or a flat folder)."""
    for d in (Path(rel), Path(rel) / 'outputs', Path(rel) / 'inputs'):
        if (d / name).exists():
            return d / name
    return Path(rel) / name


def load_umbilicus(path):
    """Control points -> (z sorted, x(z), y(z)) for linear interpolation (clamped at the ends)."""
    cp = json.loads(Path(path).read_text())['control_points']
    z = np.array([p['z'] for p in cp], float); x = np.array([p['x'] for p in cp], float)
    y = np.array([p['y'] for p in cp], float)
    o = np.argsort(z)
    return z[o], x[o], y[o]


def outward_fraction(pts, nrm, umb):
    """Fraction of valid pixels whose unit normal points away from the scroll axis (n . r_hat > 0), with r the
    horizontal vector from the umbilicus at the pixel's z to the pixel."""
    zu, xu, yu = umb
    p = pts.reshape(-1, 3); n = nrm.reshape(-1, 3)
    ok = np.isfinite(p).all(1) & np.isfinite(n).all(1)
    p, n = p[ok], n[ok]
    if not len(p):
        return float('nan'), 0
    ux = np.interp(p[:, 2], zu, xu); uy = np.interp(p[:, 2], zu, yu)
    r = np.stack([p[:, 0] - ux, p[:, 1] - uy], 1)
    r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
    dot = n[:, 0] * r[:, 0] + n[:, 1] * r[:, 1]
    return float((dot > 0).mean()), int(len(p))


def open_render(path):
    """(array, attrs) of a render zarr written by render66 (group with array '0', shape (N, H, W) uint8)."""
    import zarr
    g = zarr.open_group(str(path), mode='r')
    return g['0'], dict(g.attrs)


def write_render(path, vol, attrs):
    import zarr
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    g = zarr.open_group(str(tmp), mode='w', zarr_format=2)
    a = g.create_array('0', shape=vol.shape, chunks=(vol.shape[0], 128, 128), dtype='uint8')
    a[:] = vol
    g.attrs.update(attrs)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    os.replace(tmp, path)


def valid_mask_all_layers(arr, block_rows=256, k_range=None):
    """His ensemble validity: CT > 0 in every rendered layer (k_range = attrs rendered_k when only some were)."""
    n, H, W = arr.shape
    k0, k1 = k_range or (0, n)
    v = np.zeros((H, W), bool)
    for r0 in range(0, H, block_rows):
        v[r0:r0 + block_rows] = (np.asarray(arr[k0:k1, r0:r0 + block_rows]) > 0).all(0)
    return v
