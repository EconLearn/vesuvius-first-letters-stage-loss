"""Data access for the experiment scripts: local mirror of the manifest, zarr levels, label transfer.

Local layout (see vcpaths.py): $VC_DATA/s3/<s3 key>, $VC_DATA/hf/<repo>.
Label transfer between canvases follows AndreasHad04 bench/zs_fetch.py (MIT): nearest-neighbour
resampling of level-2 labels by a pure scale, plus an integer shift from a pre-registered
normalised-cross-correlation (Z-ALIGN) search of high-passed window-mean images within +-8 px.
"""
from __future__ import annotations
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)

import json
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter

ROOT = Path(_P.ROOT)
DATA = Path(_P.DATA)
S3 = DATA / "s3"
HTTP = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SEARCH, PEAK_TOL, PEAK_RATIO = 8, 2, 3.0

P841 = {"w00": "20260220213127-w00", "ag144": "20260220214732-auto_grown_20260220144552896",
        "ag174": "20260221022814-auto_grown_20260220174252405"}
V841_9 = "9.366um-1.2m-113keV-volume-20250821151531"
V841_2 = "2.403um-0.22m-77keV-volume-20260319124803"
LAB841 = "ink-labels/2.403um-volume-20260319124803/20260918"
R1_WORK = Path(_P.WORK) / "r1_armz"   # R1 arm A writes data/zs/<seg>/{label,support}.npy + meta.json here


def s3_local(key, allow_stream=False):
    p = S3 / key.strip("/")
    if p.exists():
        return str(p)
    if allow_stream:
        return HTTP + key.strip("/")
    raise FileNotFoundError(f"{p} not found locally: fetch it into $VC_DATA/s3 first (README: data)")


def open_level(path, level):
    """Array at pyramid `level` from an OME-Zarr group, or the array itself."""
    node = zarr.open(path, mode="r")
    if hasattr(node, "shape"):
        return node
    return node[str(level)]


def read_labels_level(labels_dir, level=2):
    ink = np.asarray(open_level(f"{labels_dir}/inklabels.zarr", level)[:]).squeeze()
    sup = np.asarray(open_level(f"{labels_dir}/supervision.zarr", level)[:]).squeeze()
    if ink.ndim == 3:
        ink = ink.any(axis=0)
    if sup.ndim == 3:
        sup = sup.any(axis=0)
    return ink > 0, sup > 0


def hp(img):
    x = img.astype(np.float64); x = x - gaussian_filter(x, 8)
    return (x - x.mean()) / (x.std() + 1e-9)


def ncc_grid(A, B, valid):
    out = np.full((2 * SEARCH + 1, 2 * SEARCH + 1), np.nan); H, W = A.shape
    for a, dy in enumerate(range(-SEARCH, SEARCH + 1)):
        for b, dx in enumerate(range(-SEARCH, SEARCH + 1)):
            ys, ye, xs, xe = max(0, -dy), min(H, H - dy), max(0, -dx), min(W, W - dx)
            a_ = A[ys:ye, xs:xe]; b_ = B[ys + dy:ye + dy, xs + dx:xe + dx]
            v = valid[ys:ye, xs:xe] & valid[ys + dy:ye + dy, xs + dx:xe + dx]
            if v.sum() > 1000:
                out[a, b] = float(np.corrcoef(a_[v], b_[v])[0, 1])
    return out


def resample(img, i0, i1, j0, j1, y0, x0, S, dy=0, dx=0, fill=0):
    """img on the reference level-2 crop (origin y0, x0) -> target crop rows [i0,i1) cols [j0,j1).
    S = reference pixels per target pixel; (dy, dx) shift in target pixels. Nearest neighbour."""
    ii = np.rint((np.arange(i0, i1) + dy + 0.5) * S - 0.5).astype(int) - y0
    jj = np.rint((np.arange(j0, j1) + dx + 0.5) * S - 0.5).astype(int) - x0
    okr, okc = (ii >= 0) & (ii < img.shape[0]), (jj >= 0) & (jj < img.shape[1])
    out = np.full((len(ii), len(jj)), fill, dtype=img.dtype)
    out[np.ix_(okr, okc)] = img[np.ix_(ii[okr], jj[okc])]
    return out, np.outer(okr, okc)


def zalign(target_mean, ref_mean_on_target, inb):
    """Pre-registered Z-ALIGN rule (ARM Z): zero offset if the NCC peak is within 2 px of zero, the peak
    shift if peak >= 3 x median |NCC|, else None (exclude)."""
    g = ncc_grid(hp(target_mean), hp(ref_mean_on_target), inb)
    pk = np.unravel_index(np.nanargmax(g), g.shape); dy, dx = pk[0] - SEARCH, pk[1] - SEARCH
    peak, med = float(g[pk]), float(np.nanmedian(np.abs(g)))
    if abs(dy) <= PEAK_TOL and abs(dx) <= PEAK_TOL:
        use = (0, 0)
    elif peak >= PEAK_RATIO * med:
        use = (int(dy), int(dx))
    else:
        use = None
    return dict(peak_shift=[int(dy), int(dx)], peak_ncc=peak, median_abs_ncc=med, used_offset=use)


def r1_canvas(seg):
    """The 841 eligible-canvas crop, labels and support produced by R1 arm A (gate must have passed)."""
    d = R1_WORK / "data" / "zs" / seg
    gate = Path(_P.OUT) / "r1" / "gate.json"
    if not gate.exists() or not json.load(open(gate)).get("gate_pass"):
        raise SystemExit("R1 gate has not passed (out/r1/gate.json): run the PHerc0841 R1 bench first (not included here)")
    meta = json.load(open(d / "meta.json"))
    return meta, np.load(d / "label.npy"), np.load(d / "support.npy")


def r1_level2(seg):
    """ARM Y level-2 crop info (2.403 um grid / 4 = 9.612 um px): meta incl. bbox_level2, labels, support."""
    d = R1_WORK / "data" / "ys" / seg
    return json.load(open(d / "meta.json")), np.load(d / "label.npy"), np.load(d / "support.npy")


def read_841_eligible_crop(seg, meta, z0=None, z1=None):
    """(D,H,W) uint8 from the local 9.366 um surface volume at the R1 crop (all 28 layers by default)."""
    a = open_level(s3_local(f"PHerc0841/segments/{P841[seg]}/surface-volumes/{V841_9}.zarr"), 0)
    i0, i1, j0, j1 = meta["crop"]
    z0 = 0 if z0 is None else z0; z1 = a.shape[0] if z1 is None else z1
    return np.asarray(a[z0:z1, i0:i1, j0:j1])


def central(D, n):
    s = max(0, (D - n) // 2)
    return s, s + n


def block_mean(img, f):
    H, W = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:H, :W].astype(np.float32).reshape(H // f, f, W // f, f).mean(axis=(1, 3))
