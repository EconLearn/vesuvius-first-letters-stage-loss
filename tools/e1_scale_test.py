#!/usr/bin/env python
"""E1 - input-scale exactness test: ink_9um on 8.64 um First-Letters-type volumes.

Pre-registration and results: summarised in SUBMISSION.md (the pre-registration was written before any arm was run).
In this repository the 1447 targets (used by g12_g2.py) and the 0343P label registration (stage labels343) are the
reproducible parts; the 0009B target needs v2_same_session.py and the 0841 control needs the R1 bench (not included).

Arms (identical crops, labels, support masks, scorer; only the input sampling differs):
  A    native 8.64 um surface volume (official behaviour: villa infer does no physical rescaling)
  B    volume resampled by f = 8.64/9.362 in H, W AND depth (anti-aliased linear), prediction mapped back
  S1   SABOTAGE: wrong factor 1/f (must score below A)
  B96  EXPLORATORY: f = 8.64/9.6 (24 of 29 ink_9um training representations are ~9.6 um)
  RT   CONTROL: f then 1/f (interpolation blur only, scale unchanged)
  C    v8-in base, native, both depth orders (1447: released predictions/v8in.png)
Checkpoints: ink_9um seed42 step-075000 (primary), seed43 step-050000 (replication).

usage: e1_scale_test.py --stage {prep,infer,v8in,score} [--targets ...] [--arms ...]
"""
from __future__ import annotations
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)

import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np, cv2, tifffile, zarr
from scipy import sparse

sys.path.insert(0, _P.TOOLS)
from vlib import metrics as M, vio, guards  # noqa: E402

V = Path(_P.ROOT); WORK = Path(_P.WORK) / "e1"; OUT = Path(_P.OUT) / "e1"
PY = sys.executable
CKPT = {"s42": Path(_P.CKPT) / "ink_9um/hybrid_3d2d-seed42/step-075000.pth",
        "s43": Path(_P.CKPT) / "ink_9um/hybrid_3d2d-seed43/step-050000.pth"}
F_B = 8.64 / 9.362
ARMS = {"A": [1.0], "B": [F_B], "S1": [1.0 / F_B], "B96": [8.64 / 9.6], "RT": [F_B, 1.0 / F_B],
        "F085": [0.85], "F117": [1.17]}   # F085/F117: exploratory scale sweep (dose-response), not part of PASS
P1447 = Path(_P.DATA) / "hf" / "ink-8um-pherc1447-surfaces"
P343_LAB = Path(_P.DATA) / "hf_bucket" / "scrollprize_datasets" / "ink" / "unused" / "P343p"
S0009B = "20250919125754-auto_grown_20250919055754487_inp_hr"
V0009B_8 = "8.64um-1.2m-116keV-volume-20250521125136"
V0343P_8 = "8.64um-1.2m-116keV-volume-20250521134555"
PIX = 8.64


# ----------------------------------------------------------------------------------------------- resampling
def axis_positions(n_in, f, centered):
    """Output sample positions (in input pixel coordinates) along one axis for size factor f."""
    if centered:  # depth: hold the centre layer fixed; same parity as n_in so the 17-layer centre crop stays centred
        c = (n_in - 1) / 2.0
        n_out = int(np.floor(2 * c * f + 1e-9)) + 1
        if (n_out - n_in) % 2:
            n_out -= 1
        return c + (np.arange(n_out) - (n_out - 1) / 2.0) / f
    n_out = int(round(n_in * f))
    return (np.arange(n_out) + 0.5) / f - 0.5


def interp_matrix(pos, n_in, f):
    """Sparse (len(pos) x n_in) anti-aliased linear interpolation: triangle kernel of half-width max(1, 1/f)."""
    hw = max(1.0, 1.0 / f)
    rows, cols, vals = [], [], []
    for k, x in enumerate(np.clip(pos, 0, n_in - 1)):
        i = np.arange(int(np.floor(x - hw)) + 1, int(np.ceil(x + hw)))
        i = i[(i >= 0) & (i < n_in)]
        w = np.maximum(0.0, 1.0 - np.abs(i - x) / hw)
        if w.sum() <= 0:
            i, w = np.array([int(round(x))]), np.array([1.0])
        rows += [k] * len(i); cols += list(i); vals += list(w / w.sum())
    return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(pos), n_in))


def resample_volume(vol, f, strip=256):
    """(D,H,W) uint8 -> resampled uint8 by size factor f in all three axes. Processed in output-row strips."""
    D, H, W = vol.shape
    pd, ph, pw = axis_positions(D, f, True), axis_positions(H, f, False), axis_positions(W, f, False)
    Rd, Rh, Rw = interp_matrix(pd, D, f), interp_matrix(ph, H, f), interp_matrix(pw, W, f)
    RwT = Rw.T.tocsr()
    out = np.empty((len(pd), len(ph), len(pw)), np.uint8)
    for r0 in range(0, len(ph), strip):
        r1 = min(len(ph), r0 + strip)
        sub = Rh[r0:r1]
        cols = sub.indices; a, b = int(cols.min()), int(cols.max()) + 1
        sub = sub[:, a:b]
        block = vol[:, a:b, :].astype(np.float32)                          # D x rows x W
        tmp = np.stack([(sub @ block[d]) @ RwT for d in range(D)])         # D x strip x W_out (dense)
        tmp = (Rd @ tmp.reshape(D, -1)).reshape(len(pd), r1 - r0, len(pw))
        out[:, r0:r1] = np.clip(np.rint(tmp), 0, 255).astype(np.uint8)
    return out, dict(f=f, depth_in=D, depth_out=len(pd), depth_pos=[float(pd[0]), float(pd[-1])], hw_out=[len(ph), len(pw)])


def map_back(pred, H, W, f):
    """Prediction on a grid resampled by size factor f -> the original (H, W) grid (exact inverse map)."""
    Hs, Ws = pred.shape
    py = (np.arange(H) + 0.5) * f - 0.5; px = (np.arange(W) + 0.5) * f - 0.5
    Ry, Rx = interp_matrix(py, Hs, 1.0 / f), interp_matrix(px, Ws, 1.0 / f)
    return np.asarray((Ry @ pred.astype(np.float32)) @ Rx.T.tocsr(), np.float32)


def self_test():
    """Known-answer checks: identity, round trip, pattern scale."""
    rng = np.random.default_rng(0)
    v = rng.integers(0, 255, (31, 200, 300)).astype(np.uint8)
    same, info = resample_volume(v, 1.0)
    ok_id = np.array_equal(same, v)
    # a horizontal sinusoid of period 40 px must come out with period 40*f
    x = np.arange(600); img = (127 + 100 * np.sin(2 * np.pi * x / 40.0))[None, None, :].repeat(31, 0).repeat(64, 1)
    r, _ = resample_volume(img.astype(np.uint8), F_B)
    spec = np.abs(np.fft.rfft(r[15, 32].astype(float) - r[15, 32].mean()))
    period = r.shape[2] / np.argmax(spec)
    # depth: a ramp z -> centre layer value preserved
    ramp = (np.arange(31) * 8).astype(np.uint8)[:, None, None].repeat(50, 1).repeat(50, 2)
    rr, ri = resample_volume(ramp, F_B)
    centre_ok = abs(float(rr[rr.shape[0] // 2, 25, 25]) - float(ramp[15, 25, 25])) <= 1
    # map_back of a prediction that is a smooth function of position is near-exact
    yy, xx = np.mgrid[0:400, 0:500]
    p = (0.5 + 0.4 * np.sin(yy / 37.0) * np.cos(xx / 53.0)).astype(np.float32)
    ps = cv2.resize(p, (int(round(500 * F_B)), int(round(400 * F_B))), interpolation=cv2.INTER_LINEAR)
    back = map_back(ps, 400, 500, F_B)
    err = float(np.abs(back - p)[5:-5, 5:-5].max())
    out = dict(identity=bool(ok_id), period_expected=40 * F_B, period_measured=float(period),
               period_ok=bool(abs(period - 40 * F_B) < 0.5), depth_out=ri["depth_out"], centre_ok=bool(centre_ok),
               map_back_max_err=err, map_back_ok=bool(err < 0.01))
    out["all_ok"] = bool(out["identity"] and out["period_ok"] and out["centre_ok"] and out["map_back_ok"])
    return out


# ----------------------------------------------------------------------------------------------- targets
def bbox(mask, margin, shape):
    ys, xs = np.nonzero(mask)
    return (max(0, int(ys.min()) - margin), min(shape[0], int(ys.max()) + 1 + margin),
            max(0, int(xs.min()) - margin), min(shape[1], int(xs.max()) + 1 + margin))


def load_1447(w):
    d = P1447 / w
    files = sorted((d / "layers").glob("*.tif"), key=lambda p: int(p.stem))
    lab = cv2.imread(str(d / "labels" / "inklabels.png"), cv2.IMREAD_GRAYSCALE) > 127
    msk = cv2.imread(str(d / "labels" / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    i0, i1, j0, j1 = bbox(msk, 160, lab.shape)
    layers = []
    for f in files:
        a = tifffile.imread(f)
        if a.dtype != np.uint8:
            raise SystemExit(f"{f}: dtype {a.dtype}; expected uint8")
        layers.append(a[i0:i1, j0:j1])
    vol = np.stack(layers)
    rel = cv2.imread(str(d / "predictions" / "v8in.png"), cv2.IMREAD_GRAYSCALE)
    extra = {"v8in_released": rel[i0:i1, j0:j1]} if rel is not None else {}
    return vol, lab[i0:i1, j0:j1] & msk[i0:i1, j0:j1], msk[i0:i1, j0:j1], dict(crop=[i0, i1, j0, j1], layers=len(files)), extra


def load_0009B():
    # needs v2_same_session.py (0009B label transfer; not included in this repository)
    sys.path.insert(0, _P.TOOLS)
    import v2_same_session as V2
    meta, lab, sup = V2.labels_0009B()
    i0, i1, j0, j1 = meta["crop"]
    a8 = V2.sv("PHerc0009B", S0009B, V0009B_8)
    vol = np.asarray(a8[:, i0:i1, j0:j1])
    return vol, lab & sup, sup, meta, {}


def _arr(path, level):
    """Level array of a pyramid stored as bare numbered arrays (no .zgroup), or of an OME group."""
    p = Path(path) / str(level)
    return zarr.open_array(str(p), mode="r") if p.exists() else zarr.open(str(path), mode="r")[str(level)]


def _match(T, tmpl, cy, cx, search):
    """Best TM_CCOEFF_NORMED location of tmpl (centred at cy, cx in T's frame if unshifted) within +-search px."""
    h, w = tmpl.shape
    y0, x0 = cy - h // 2 - search, cx - w // 2 - search
    if y0 < 0 or x0 < 0 or y0 + h + 2 * search > T.shape[0] or x0 + w + 2 * search > T.shape[1]:
        return None
    res = cv2.matchTemplate(T[y0:y0 + h + 2 * search, x0:x0 + w + 2 * search], tmpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    med = float(np.median(np.abs(res)))
    return dict(dy=loc[1] - search, dx=loc[0] - search, peak=float(mx), ratio=float(mx / max(med, 1e-6)))


def stage_labels343():
    """Map 0343P 'unused' labels (2.215 um render grid, level 2 ~ 8.86 um) onto each segment's 8.64 um canvas.
    Registration: reference = mean over all layers of the label-grid render (level 2); target = mean of the
    central 17 layers of the 8.64 um volume (same physical depth, +-70 um). Global shift by template matching of
    the central half, then a 7x5 grid of 320 px tiles (+-40 px search) gives point correspondences; an affine
    map target->reference is fitted by least squares. Accepted only if >= 8 tiles pass (peak >= 0.3, ratio >= 3)
    and the affine residual RMS <= 1.5 target px (tiles: peak >= 0.1, peak/median >= 5, band-passed). Labels are moved by nearest neighbour. Unmatched = reported."""
    from scipy.ndimage import gaussian_filter
    (WORK / "labels").mkdir(parents=True, exist_ok=True)
    report = {}
    S3_0343P = Path(_P.DATA) / "s3" / "PHerc0343P" / "segments"
    segs = {p.name.split("-")[0]: p.name for p in S3_0343P.iterdir()}
    for d in sorted(P343_LAB.iterdir()):
        sid = d.name; rec = dict(label_dir=str(d))
        try:
            if sid not in segs:
                raise ValueError(f"no 8.64 um segment for {sid}")
            seg = segs[sid]; rec["segment"] = seg
            img2 = _arr(d / f"{sid}.zarr", 2); ink2 = _arr(d / f"{sid}_inklabels.zarr", 2); sup2 = _arr(d / f"{sid}_supervision_mask.zarr", 2)
            rec["shapes"] = dict(img_l2=list(img2.shape), ink_l2=list(ink2.shape), sup_l2=list(sup2.shape))
            ink = np.asarray(ink2[:]).squeeze(); sup = np.asarray(sup2[:]).squeeze()
            if ink.ndim == 3: ink = ink.max(0)
            if sup.ndim == 3: sup = sup.max(0)
            rec["ink_values"] = [int(v) for v in np.unique(ink)[:10]]; rec["sup_values"] = [int(v) for v in np.unique(sup)[:10]]
            ref = np.asarray(img2[:]).astype(np.float32).mean(0)
            a8 = vio.open_level(vio.s3_local(f"PHerc0343P/segments/{seg}/surface-volumes/{V0343P_8}.zarr"), 0)
            z0, z1 = vio.central(a8.shape[0], 17)
            tgt = np.asarray(a8[z0:z1]).astype(np.float32).mean(0)
            # orientation: the label-grid render need not share the 8.64 um render's handedness. Test the 8
            # flips/transposes at ~35 um (ref level 4 vs target level 2) and keep one only if it wins clearly.
            ORI = {"id": lambda x: x, "fliplr": lambda x: x[:, ::-1], "flipud": lambda x: x[::-1], "rot180": lambda x: x[::-1, ::-1],
                   "T": lambda x: x.T, "T_fliplr": lambda x: x.T[:, ::-1], "T_flipud": lambda x: x.T[::-1], "T_rot": lambda x: x.T[::-1, ::-1]}
            r4 = np.asarray(_arr(d / f"{sid}.zarr", 4)[:]).astype(np.float32).mean(0)
            t2 = np.asarray(vio.open_level(vio.s3_local(f"PHerc0343P/segments/{seg}/surface-volumes/{V0343P_8}.zarr"), 2)[z0:z1]).astype(np.float32).mean(0)
            sc = (2.215 * 16) / (8.64 * 4)
            r4 = cv2.resize(r4, (int(round(r4.shape[1] * sc)), int(round(r4.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            def _hp(x):
                y = x - cv2.GaussianBlur(x, (0, 0), 6); m = x > 0; y[~m] = 0
                return np.ascontiguousarray(((y - y[m].mean()) / (y[m].std() + 1e-6) * m).astype(np.float32))
            T2 = cv2.copyMakeBorder(_hp(t2), 150, 150, 150, 150, cv2.BORDER_CONSTANT, 0); scores = {}
            for k, f in ORI.items():
                A_ = _hp(np.ascontiguousarray(f(r4))); h_, w_ = A_.shape; th, tw = h_ // 2, w_ // 2
                tm = np.ascontiguousarray(A_[(h_ - th) // 2:(h_ - th) // 2 + th, (w_ - tw) // 2:(w_ - tw) // 2 + tw])
                if tm.shape[0] > T2.shape[0] or tm.shape[1] > T2.shape[1]:
                    continue
                scores[k] = float(cv2.minMaxLoc(cv2.matchTemplate(T2, tm, cv2.TM_CCOEFF_NORMED))[1])
            best = sorted(scores, key=scores.get, reverse=True)
            rec["orientation_scores"] = scores; rec["orientation"] = best[0]
            if scores[best[0]] < 3 * scores[best[1]]:
                raise ValueError(f"orientation ambiguous: {scores}")
            f = ORI[best[0]]
            ref, ink, sup = np.ascontiguousarray(f(ref)), np.ascontiguousarray(f(ink)), np.ascontiguousarray(f(sup))
            S = 8.64 / (2.215 * 4)                                  # reference px per target px (nominal)
            H, W = tgt.shape
            B, inb = vio.resample(ref, 0, H, 0, W, 0, 0, S, fill=0.0)
            hpT, hpB = vio.hp(gaussian_filter(tgt, 1.5)).astype(np.float32), vio.hp(gaussian_filter(B, 1.5)).astype(np.float32)  # band-pass
            hpT[tgt == 0] = 0; hpB[~inb | (B == 0)] = 0
            # global shift: central half of B searched within +-64 px in T
            h2, w2 = H // 2, W // 2
            g = _match(hpT, np.ascontiguousarray(hpB[H // 4:H // 4 + h2, W // 4:W // 4 + w2]), H // 2, W // 2, 64)
            rec["global"] = g
            if g is None or g["ratio"] < 5 or g["peak"] < 0.05:
                raise ValueError(f"global registration failed: {g}")
            pts_t, pts_r, tiles = [], [], []
            for cy in np.linspace(H * 0.1, H * 0.9, 7).astype(int):
                for cx in np.linspace(W * 0.1, W * 0.9, 5).astype(int):
                    t = hpB[cy - 160:cy + 160, cx - 160:cx + 160]
                    if t.shape != (320, 320) or (t == 0).mean() > 0.3:
                        continue
                    m = _match(hpT, np.ascontiguousarray(t), cy + g["dy"], cx + g["dx"], 40)
                    if m is None:
                        continue
                    dy, dx = g["dy"] + m["dy"], g["dx"] + m["dx"]
                    tiles.append(dict(cy=int(cy), cx=int(cx), dy=int(dy), dx=int(dx), peak=m["peak"], ratio=m["ratio"]))
                    if m["peak"] >= 0.1 and m["ratio"] >= 5:
                        pts_t.append([cy + dy, cx + dx, 1.0]); pts_r.append([S * (cy + 0.5) - 0.5, S * (cx + 0.5) - 0.5])
            rec["tiles"] = tiles
            if len(pts_t) < 8:
                raise ValueError(f"only {len(pts_t)} good tiles")
            Pt, Pr = np.asarray(pts_t), np.asarray(pts_r); keep = np.ones(len(Pt), bool)
            while True:   # least-squares affine with iterative rejection of the worst tile while it is > 3 px off
                A, *_ = np.linalg.lstsq(Pt[keep], Pr[keep], rcond=None)   # (3, 2): r = [y x 1] @ A
                err = np.sqrt(((Pt @ A - Pr) ** 2).sum(1)) / S
                worst = int(np.argmax(np.where(keep, err, -1)))
                if err[worst] <= 3.0 or keep.sum() <= 8:
                    break
                keep[worst] = False
            rms = float(np.sqrt((err[keep] ** 2).mean()))
            rec["affine"] = A.tolist(); rec["affine_rms_target_px"] = rms; rec["n_tiles_used"] = int(keep.sum())
            rec["tiles_rejected"] = [int(i) for i in np.nonzero(~keep)[0]]
            if keep.sum() < 8:
                raise ValueError(f"only {int(keep.sum())} inlier tiles")
            # The global affine is reported but NOT used for the labels when it fails (the two renders differ by a
            # non-affine warp: measured). Labels are placed per labelled component by a LOCAL translation: a 320 px
            # band-passed template around the component in B, matched in T within +-80 px of the global shift.
            from scipy import ndimage as ndi
            thr_i = 0.5 * float(ink.max()) if ink.max() > 0 else 1
            labB, _ = vio.resample((ink >= thr_i).astype(np.uint8), 0, H, 0, W, 0, 0, S, fill=0)
            labB = labB.astype(bool)
            cc, ncc_ = ndi.label(ndi.binary_dilation(labB, iterations=24))   # group strokes of one letter/component
            lab = np.zeros((H, W), bool); comps = []
            for i, sl in enumerate(ndi.find_objects(cc), start=1):
                cy, cx = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
                ty0, tx0 = max(0, cy - 160), max(0, cx - 160)
                t = hpB[ty0:ty0 + 320, tx0:tx0 + 320]
                m = _match(hpT, np.ascontiguousarray(t), ty0 + t.shape[0] // 2 + g["dy"], tx0 + t.shape[1] // 2 + g["dx"], 80) if t.size else None
                comp = dict(id=i, bbox=[sl[0].start, sl[0].stop, sl[1].start, sl[1].stop], px=int((labB & (cc == i)).sum()))
                if m is None or m["peak"] < 0.1 or m["ratio"] < 5:
                    comp.update(ok=False, match=m); comps.append(comp); continue
                dy, dx = g["dy"] + m["dy"], g["dx"] + m["dx"]
                comp.update(ok=True, dy=int(dy), dx=int(dx), peak=m["peak"], ratio=m["ratio"])
                src = labB & (cc == i)
                ys, xs = np.nonzero(src); ys2, xs2 = ys + dy, xs + dx
                okp = (ys2 >= 0) & (ys2 < H) & (xs2 >= 0) & (xs2 < W)
                lab[ys2[okp], xs2[okp]] = True; comps.append(comp)
            rec["components"] = comps; rec["components_used"] = int(sum(c["ok"] for c in comps))
            rec["affine_ok"] = bool(rms <= 1.5)
            if not lab.any():
                raise ValueError("no component registered")
            rec["supervision_equals_labels"] = bool(np.array_equal(ink >= thr_i, sup >= (0.5 * float(sup.max()) if sup.max() > 0 else 1)))
            b0, b1, c0, c1 = bbox(lab, 160, (H, W))
            su = np.zeros((H, W), bool); su[b0:b1, c0:c1] = True
            su &= tgt > 0
            rec["ncc_registered"] = None; rec["ncc_unregistered"] = None
            if not su.any() or not (lab & su).any():
                raise ValueError("empty support or no ink after mapping")
            i0, i1, j0, j1 = bbox(su, 160, (H, W))
            np.savez_compressed(WORK / "labels" / f"0343P_{sid}.npz", label=lab[i0:i1, j0:j1], support=su[i0:i1, j0:j1])
            rec.update(crop=[i0, i1, j0, j1], label_px=int((lab & su).sum()), support_px=int(su.sum()), pixel_um=8.64, ok=True)
            (WORK / "labels" / f"0343P_{sid}.json").write_text(json.dumps(rec, indent=1, default=float))
        except Exception as e:  # report, never guess
            rec.update(ok=False, error=repr(e))
        report[sid] = rec
        print(sid, {k: rec.get(k) for k in ("ok", "error", "global", "n_tiles_used", "affine_rms_target_px", "ncc_registered",
                                              "ncc_unregistered", "label_px", "support_px")}, flush=True)
    (WORK / "labels" / "0343P_mapping_report.json").write_text(json.dumps(report, indent=1, default=float))


def load_0343P(name):
    z = np.load(WORK / "labels" / f"0343P_{name}.npz")
    meta = json.loads((WORK / "labels" / f"0343P_{name}.json").read_text())
    seg = meta["segment"]; i0, i1, j0, j1 = meta["crop"]
    a = vio.open_level(vio.s3_local(f"PHerc0343P/segments/{seg}/surface-volumes/{V0343P_8}.zarr"), 0)
    vol = np.asarray(a[:, i0:i1, j0:j1])
    return vol, z["label"] & z["support"], z["support"], meta, {}


def load_841(seg):
    """Known-answer control: PHerc0841 on the R1 eligible 9.366 um canvas (labels/support/crop from R1 arm A)."""
    d = vio.R1_WORK / "data" / "zs" / seg
    meta = json.loads((d / "meta.json").read_text())
    lab, sup = np.load(d / "label.npy").astype(bool), np.load(d / "support.npy").astype(bool)
    vol = vio.read_841_eligible_crop(seg, meta)
    assert vol.shape[1:] == lab.shape, (vol.shape, lab.shape)
    return vol, lab & sup, sup, dict(crop=meta["crop"], pixel_um=9.366, r1_meta_keys=sorted(meta)), {}


def all_targets():
    t = {"1447_w058": ("1447", lambda: load_1447("w058")), "1447_w060": ("1447", lambda: load_1447("w060")),
         "0009B": ("0009B", load_0009B)}
    for seg in ("w00", "ag144", "ag174"):
        if (vio.R1_WORK / "data" / "zs" / seg / "label.npy").exists():
            t[f"0841_{seg}"] = ("0841", (lambda s=seg: load_841(s)))
    for f in sorted((WORK / "labels").glob("0343P_*.npz")):
        n = f.stem.replace("0343P_", "")
        t[f"0343P_{n}"] = ("0343P", (lambda n=n: load_0343P(n)))
    return t


def write_zarr(path, vol):
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    a = g.create_array("0", shape=vol.shape, chunks=(vol.shape[0], 128, 128), dtype="uint8")
    a[:] = vol


def stage_prep(names):
    T = all_targets()
    for name in names or T:
        if name not in T:
            print("unknown/unavailable target", name); continue
        d = WORK / name
        if (d / "prep.json").exists():
            print("have", name); continue
        d.mkdir(parents=True, exist_ok=True)
        t = time.time()
        vol, lab, sup, meta, extra = T[name][1]()
        np.savez_compressed(d / "labels.npz", label=lab, support=sup)
        for k, v in extra.items():
            cv2.imwrite(str(d / f"{k}.png"), v)
        rec = dict(scroll=T[name][0], meta=meta, shape=list(vol.shape), label_px=int(lab.sum()), support_px=int(sup.sum()),
                   pixel_um=float(meta.get("pixel_um", PIX)), arms={})
        for arm, fs in ARMS.items():
            x = vol; infos = []
            for f in fs:
                x, info = resample_volume(x, f) if f != 1.0 else (x, dict(f=1.0, depth_out=x.shape[0]))
                infos.append(info)
            write_zarr(d / f"{arm}.zarr", x)
            rec["arms"][arm] = dict(shape=list(x.shape), steps=infos)
            print(name, arm, x.shape, flush=True)
        rec["prep_secs"] = round(time.time() - t, 1)
        (d / "prep.json").write_text(json.dumps(rec, indent=1, default=float))


# ----------------------------------------------------------------------------------------------- inference
def g0_busy():
    r = subprocess.run(["pgrep", "-f", "r1_ink9_baselin[e]|r1_zs_infer_devic[e]"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_gpu(need_mb, wait_s=6 * 3600):
    t0 = time.time()
    while g0_busy():
        if time.time() - t0 > wait_s:
            raise SystemExit("G0 still running; giving up")
        print("[e1] G0 baseline running: waiting", flush=True); time.sleep(60)
    guards.gpu_check(need_mb, wait_s=wait_s)


def stage_infer(names, arms, ckpts, workers=4):
    """Official villa flat inference, called in-process (one model build per checkpoint; a fresh CLI process
    per run cost ~9 min of start-up on this shared PC, measured). Same functions main() calls:
    parse_args -> configure_model -> prepare_model_for_inference -> infer_single_zarr per direction."""
    import logging, multiprocessing as mp
    from dataclasses import replace
    mp.set_start_method("fork", force=True)   # py3.14 default forkserver re-imports torch per worker per call
    from vesuvius.ink_detection.inference import infer as INF
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    import torch
    torch.set_float32_matmul_precision("high")
    # priority phases: the pre-registered arms first, replication next, exploratory arms last
    phases = [("s42", ["A", "B", "S1"]), ("s43", ["A", "B", "S1"]), ("s42", ["RT", "B96"]), ("s42", ["F085", "F117"]),
              ("s43", ["RT", "B96"]), ("s43", ["F085", "F117"])]
    order = ["1447_w058", "1447_w060", "0009B"] + sorted(n for n in names if n.startswith("0343P")) + sorted(n for n in names if n.startswith("0841"))
    order = [n for n in order if n in names] + [n for n in names if n not in order]
    models = {}
    for ck, parms in phases:
        if ck not in ckpts:
            continue
        jobs = [(n, arm) for arm in parms if arm in arms for n in order
                if (WORK / n / "prep.json").exists() and (WORK / n / f"{arm}.zarr").exists()
                and not (WORK / n / "pred" / f"{ck}_{arm}_reverse.tif").exists()]
        if not jobs:
            continue
        if ck not in models:
            n0, a0 = jobs[0]
            (WORK / n0 / "pred").mkdir(parents=True, exist_ok=True)
            args = INF.parse_args([str(WORK / n0 / f"{a0}.zarr"), str(CKPT[ck]), str(WORK / n0 / "pred" / "x.tif"),
                                   "--direction", "both", "--batch-size", "32", "--num-workers", str(workers), "--no-compile"])
            args = INF.normalize_inference_paths(args)
            wait_gpu(2500, wait_s=6 * 3600)
            t = time.time()
            configured = INF.configure_model(args)
            model, device = INF.prepare_model_for_inference(configured.model, gpu_ids=args.gpu_ids,
                                                            compile_model=False, compile_mode=args.compile_mode)
            models[ck] = (args, replace(configured, model=model), device)
            print(f"[e1] model {ck} ready in {time.time() - t:.0f}s amp={configured.amp_dtype}", flush=True)
        args, configured, device = models[ck]
        for name, arm in jobs:
            d = WORK / name; out = d / "pred" / f"{ck}_{arm}.tif"; out.parent.mkdir(parents=True, exist_ok=True)
            wait_gpu(700, wait_s=6 * 3600)   # model already resident: this process's own cache counts as "used"
            t = time.time()
            for direction in ("forward", "reverse"):
                INF.infer_single_zarr(args=args, input_zarr=str(d / f"{arm}.zarr"), configured_model=configured, device=device,
                                      output_tiff=INF.resolve_single_output_path(out, direction=direction, requested_direction="both"),
                                      layer_direction=direction)
            print(f"[e1] {name} {ck} {arm} {time.time() - t:.0f}s", flush=True)


def stage_v8in(names, stride, bs):
    import torch
    sys.path.insert(0, _P.V8IN_DIR)
    from ink8um import InkDetector, predict_stack
    torch.cuda.set_per_process_memory_fraction(0.85)
    model = None
    for name in names:
        d = WORK / name
        if name.startswith(("1447", "0841")) or not (d / "prep.json").exists():
            continue
        for dirn, rev in (("forward", False), ("reverse", True)):
            out = d / "pred" / f"v8in_{dirn}.png"
            if out.exists():
                continue
            # this process's own cached memory counts as "used": demand the full budget only before the first load
            wait_gpu(1000 + 1000 * bs if model is None else 500)
            if model is None:
                model = InkDetector.from_pretrained(_P.V8IN_DIR).eval()
            arr = zarr.open_group(str(d / "A.zarr"), mode="r")["0"]
            z0, z1 = vio.central(arr.shape[0], 24)
            st = np.ascontiguousarray(np.clip(np.asarray(arr[z0:z1]), 0, 200).transpose(1, 2, 0))
            t = time.time()
            p = predict_stack(model, st, reverse=rev, stride=stride, batch_size=bs, num_workers=2, device="cuda")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.rint(p * 255).astype(np.uint8))
            print(f"{name} v8in {dirn} {time.time() - t:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- scoring
def load_pred(name, ck, arm, dirn):
    d = WORK / name
    rec = json.loads((d / "prep.json").read_text())
    if ck == "v8in":
        f = (d / "pred" / f"v8in_{dirn}.png") if not name.startswith("1447") else (d / "v8in_released.png")
        if not f.exists():
            return None
        return cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    f = d / "pred" / (f"{ck}_{arm}.tif" if dirn == "forward" else f"{ck}_{arm}_reverse.tif")
    if not f.exists():
        return None
    p = tifffile.imread(f).astype(np.float32)
    p = p / (255.0 if p.max() > 1.0 else 1.0)
    H, W = rec["shape"][1:]
    if arm in ("A", "RT"):
        assert p.shape == (H, W), (name, arm, p.shape, (H, W))
        return p
    f_ = ARMS[arm][0]
    return np.clip(map_back(p, H, W, f_), 0, 1)


def letters_elong(pred, label, support, pixel_um=PIX):
    """Per-letter eval_forme elongation values (same rule as vlib.metrics.elongation, kept per letter for pooling)."""
    from scipy import ndimage
    k = 9.612 / float(pixel_um)
    min_extent, min_area = 150.0 * k, 12.0 * k * k
    L = label.astype(bool) & support.astype(bool)
    lab, _ = ndimage.label(L)
    objs = [(i + 1, s) for i, s in enumerate(ndimage.find_objects(lab))
            if s is not None and max(s[0].stop - s[0].start, s[1].stop - s[1].start) >= min_extent]
    if not objs:
        return dict(map=[], labels=[], padded=[])
    t = np.quantile(np.concatenate([pred[s][lab[s] == i] for i, s in objs]), 0.30)
    return dict(map=[M._elong(pred[s] >= t, min_area) for i, s in objs],
                labels=[M._elong(lab[s] == i, min_area) for i, s in objs],
                padded=[M._elong(pred[s] >= t, min_area, pad=True) for i, s in objs])


def paired_delta_ci(hx, hy, n=1000, seed=0):
    """Paired 64 px block bootstrap of AUC(y) - AUC(x); hx, hy = (pos, neg) per-block histograms on the same blocks."""
    rng = np.random.default_rng(seed); k = len(hx[0]); out = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        out.append(M.auc_from(hy[0][i].sum(0), hy[1][i].sum(0)) - M.auc_from(hx[0][i].sum(0), hx[1][i].sum(0)))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def stage_score(arms, ckpts):
    T = all_targets()
    names = [n for n in T if (WORK / n / "prep.json").exists()]
    res = dict(targets={}, scrolls={}, settings=dict(arms={k: v for k, v in ARMS.items()}, ckpts={k: str(v) for k, v in CKPT.items()}))
    hists = {}   # (scroll, ck, arm) -> list of (pos, neg) block hist arrays (direction = arm-A label-free choice)
    letters = {}
    for name in names:
        scroll = T[name][0]
        z = np.load(WORK / name / "labels.npz"); lab, sup = z["label"], z["support"]
        pix = float(json.loads((WORK / name / "prep.json").read_text()).get("pixel_um", PIX))
        tr = dict(scroll=scroll, label_px=int(lab.sum()), support_px=int(sup.sum()), ink_rate=float(lab[sup].mean()), ck={})
        for ck in list(ckpts) + ["v8in"]:
            ckrec = {}
            arm_list = arms if ck != "v8in" else ["C"]
            for arm in arm_list:
                row = {}
                for dirn in ("forward", "reverse"):
                    if ck == "v8in" and name.startswith("1447") and dirn == "reverse":
                        continue
                    p = load_pred(name, ck, arm, dirn)
                    if p is None:
                        continue
                    row[dirn] = dict(auc=M.auc(p, lab, sup), sep=M.separation(p, sup))
                if row:
                    ckrec[arm] = row
            if not ckrec:
                continue
            # depth order: label-free rule on arm A (v8in: on itself); 1447 released v8in map is already oriented
            base = ckrec.get("A", ckrec.get("C"))
            if "reverse" in base:
                chosen = M.choose_direction(base["forward"]["sep"], base["reverse"]["sep"])
            else:
                chosen = "forward"
            ckrec["chosen_direction"] = chosen
            ckrec["label_best_direction_A"] = (max(("forward", "reverse"), key=lambda k: base[k]["auc"]) if "reverse" in base else "forward")
            for arm in arm_list:
                if arm not in ckrec or chosen not in ckrec[arm]:
                    continue
                p = load_pred(name, ck, arm, chosen)
                r = ckrec[arm]; r["auc"] = r[chosen]["auc"]; r["ci95"] = M.auc_ci(p, lab, sup, n=300)
                le = letters_elong(p, lab, sup, pix)
                r["elong_map"] = float(np.nanmedian(le["map"])) if le["map"] else float("nan")
                r["elong_labels"] = float(np.nanmedian(le["labels"])) if le["labels"] else float("nan")
                r["elong_padded"] = float(np.nanmedian(le["padded"])) if le["padded"] else float("nan")
                r["letters"] = len(le["map"])
                letters.setdefault((scroll, ck, arm), []).append(le)
                pos, neg = M.block_hists(M.to_u8(p), lab.astype(bool), sup.astype(bool))
                hists.setdefault((scroll, ck, arm), []).append((pos, neg))
                if arm in ("A", "B", "C"):
                    rng = np.random.default_rng(7)
                    sl = lab[sup].copy(); rng.shuffle(sl); ls = np.zeros_like(lab); ls[sup] = sl
                    r["S2_pixel_shuffled_auc"] = M.auc(p, ls, sup)
                    r["S2_c2_floor_mean"] = float(np.mean(M.shuffle_floor(p, lab, sup, seeds=5)))
            tr["ck"][ck] = ckrec
        res["targets"][name] = tr
        print(name, json.dumps({ck: {a: round(v["auc"], 4) for a, v in r.items() if isinstance(v, dict) and "auc" in v}
                                for ck, r in tr["ck"].items()}), flush=True)
    # per-scroll pooled AUC, paired delta CIs, pooled elongation
    for (scroll, ck, arm), hl in hists.items():
        pos = np.concatenate([h[0] for h in hl]); neg = np.concatenate([h[1] for h in hl])
        s = res["scrolls"].setdefault(scroll, {}).setdefault(ck, {})
        le = letters[(scroll, ck, arm)]
        lm = sum((x["map"] for x in le), []); ll = sum((x["labels"] for x in le), []); lp = sum((x["padded"] for x in le), [])
        s[arm] = dict(auc=M.auc_from(pos.sum(0), neg.sum(0)), windows=len(hl), letters=len(lm),
                      elong_map=float(np.nanmedian(lm)) if lm else float("nan"),
                      elong_labels=float(np.nanmedian(ll)) if ll else float("nan"),
                      elong_padded=float(np.nanmedian(lp)) if lp else float("nan"))
        s[arm]["_h"] = (pos, neg)
    for scroll, byck in res["scrolls"].items():
        for ck, s in byck.items():
            if "A" not in s:
                continue
            for arm in s:
                if arm == "A" or not isinstance(s[arm], dict) or "_h" not in s[arm]:
                    continue
                s[arm]["delta_vs_A"] = s[arm]["auc"] - s["A"]["auc"]
                s[arm]["delta_vs_A_ci95"] = paired_delta_ci(s["A"]["_h"], s[arm]["_h"])
                s[arm]["elong_delta_vs_A"] = s[arm]["elong_map"] - s["A"]["elong_map"]
    for scroll, byck in res["scrolls"].items():
        for ck, s in byck.items():
            for arm in s:
                s[arm].pop("_h", None)
    # verdict
    def verdict(ck):
        rows = {sc: res["scrolls"][sc].get(ck, {}) for sc in res["scrolls"] if sc in ("1447", "0009B", "0343P")}
        passing = [sc for sc, s in rows.items() if "B" in s and s["B"].get("delta_vs_A", -1) >= 0.02
                   and s["B"]["elong_map"] >= s["A"]["elong_map"] - 1.0]
        s1_loses = [sc for sc, s in rows.items() if "S1" in s and s["S1"].get("delta_vs_A", 1) < 0]
        return dict(scrolls=len(rows), B_pass_scrolls=passing, S1_loses_scrolls=s1_loses,
                    pass_=len(passing) >= 2, s1_ok=len(s1_loses) >= 2)
    res["verdict"] = {ck: verdict(ck) for ck in ckpts}
    v42, v43 = res["verdict"].get("s42", {}), res["verdict"].get("s43", {})
    res["PASS"] = bool(v42.get("pass_") and v43.get("pass_") and v42.get("s1_ok") and v43.get("s1_ok"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    # markdown
    L = ["| scroll | ckpt | arm | AUC (pooled) | dAUC vs A [paired 95% CI] | elong map / labels | letters |", "|---|---|---|---|---|---|---|"]
    for scroll, byck in res["scrolls"].items():
        for ck, s in byck.items():
            for arm, r in s.items():
                ci = r.get("delta_vs_A_ci95")
                dd = f"{r['delta_vs_A']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "-"
                L.append(f"| {scroll} | {ck} | {arm} | {r['auc']:.4f} | {dd} | {r['elong_map']:.1f} / {r['elong_labels']:.1f} | {r['letters']} |")
    L.append("\nverdict: " + json.dumps(res["verdict"]) + f"\nPASS: {res['PASS']}")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["selftest", "labels343", "prep", "infer", "v8in", "score"])
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--ckpts", nargs="*", default=["s42", "s43"])
    ap.add_argument("--stride", type=int, default=21)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "selftest":
        r = self_test(); print(json.dumps(r, indent=1)); OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "resample_selftest.json").write_text(json.dumps(r, indent=1)); sys.exit(0 if r["all_ok"] else 1)
    guards.require_terms(); guards.renice()
    WORK.mkdir(parents=True, exist_ok=True)
    names = a.targets or [n for n in all_targets()]
    if a.stage == "labels343":
        stage_labels343()
    elif a.stage == "prep":
        stage_prep(a.targets)
    elif a.stage == "infer":
        stage_infer(names, a.arms, a.ckpts)
    elif a.stage == "v8in":
        stage_v8in(names, a.stride, a.batch_size)
    elif a.stage == "score":
        stage_score(a.arms, a.ckpts)


if __name__ == "__main__":
    main()
