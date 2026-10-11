"""Scoring used by every arm, so all arms are measured on the same pixels the same way.

* AUC: exact P(score_pos > score_neg) + 0.5 P(tie) from 256-bin histograms of uint8 maps, inside the
  support (supervision) mask. Same definition as AndreasHad04/villa-apple-silicon bench/xs_score.py
  (used for the #1867/#1898 numbers), re-implemented here so the two scorers can be cross-checked.
* 64 px block bootstrap CI (blocks resampled with replacement).
* Label-free depth-direction rule ("separation"): mean(p | p > .5) - mean(p | p <= .5); larger wins.
* Nulls: label roll by half the width (must give 0.50 +- 0.03) and the C2 block-derangement floor.
* Elongation: liliandevarieux/vesuvius-841 scripts/eval_forme.py (PR-32) formula, generalised to any
  pixel size: letters = labelled components with bbox extent >= 150 px at 9.612 um (1.44 mm);
  threshold = 30th percentile of the map under all letters; per letter the area-weighted median of
  area / r^2 (r = max inscribed radius) over thresholded components >= 12 px (at 9.612 um);
  report the median over letters. Human tracings score 16-25; held-out blobs ~11 (#1907).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BLOCK = 64


def to_u8(pred):
    p = np.asarray(pred)
    if p.dtype == np.uint8:
        return p
    p = p.astype(np.float64)
    if p.max() > 1.0 + 1e-6:  # uint16 / 0..65535 or 0..255 floats
        p = p / (65535.0 if p.max() > 255.0 else 255.0)
    return np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8)


def block_hists(pred_u8, label, support, block=BLOCK):
    H, W = label.shape
    by, bx = np.arange(H) // block, np.arange(W) // block
    bid = (by[:, None] * (bx.max() + 1) + bx[None, :])[support]
    v = pred_u8[support].astype(np.int64); y = label[support].astype(bool)
    nb = int((by.max() + 1) * (bx.max() + 1))
    pos = np.bincount(bid[y] * 256 + v[y], minlength=nb * 256).reshape(nb, 256)
    neg = np.bincount(bid[~y] * 256 + v[~y], minlength=nb * 256).reshape(nb, 256)
    keep = (pos.sum(1) + neg.sum(1)) > 0
    return pos[keep], neg[keep]


def auc_from(pos, neg):
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    below = np.concatenate([[0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (npos * nneg))


def auc(pred, label, support):
    pos, neg = block_hists(to_u8(pred), label.astype(bool), support.astype(bool))
    return auc_from(pos.sum(0), neg.sum(0))


def auc_ci(pred, label, support, n=1000, seed=0):
    pos, neg = block_hists(to_u8(pred), label.astype(bool), support.astype(bool))
    rng = np.random.default_rng(seed); k = len(pos)
    vals = [auc_from(pos[i].sum(0), neg[i].sum(0)) for i in (rng.integers(0, k, k) for _ in range(n))]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def separation(pred, support):
    p = to_u8(pred)[support.astype(bool)].astype(np.float64) / 255.0
    hi, lo = p[p > 0.5], p[p <= 0.5]
    return float(hi.mean() - lo.mean()) if hi.size and lo.size else float("nan")


def choose_direction(sep_fwd, sep_rev):
    """ARM X pre-registered label-free rule: larger separation wins; nan never wins silently."""
    if np.isnan(sep_fwd) and np.isnan(sep_rev):
        return None
    if np.isnan(sep_fwd):
        return "reverse"
    if np.isnan(sep_rev):
        return "forward"
    return "forward" if sep_fwd >= sep_rev else "reverse"


def label_roll_null(pred, label, support):
    """Labels AND support rolled by half the width: a real scorer must return 0.50 +- 0.03."""
    s = label.shape[1] // 2
    return auc(pred, np.roll(label, s, axis=1), np.roll(support, s, axis=1) & support)


def shuffle_floor(pred, label, support, seeds=20, min_px=32, block=BLOCK):
    """C2 floor (AndreasHad04): each block's labels drawn from a different block (derangement)."""
    pred = to_u8(pred); label = label.astype(bool); support = support.astype(bool)
    H, W = label.shape
    bid = (np.arange(H)[:, None] // block) * ((W - 1) // block + 1) + (np.arange(W)[None, :] // block)
    b, y, v = bid[support], label[support], pred[support].astype(np.int64)
    ids, counts = np.unique(b, return_counts=True); ok = ids[counts >= min_px]
    keep = np.isin(b, ok); b, y, v = b[keep], y[keep], v[keep]
    order = np.argsort(b, kind="stable"); b, y, v = b[order], y[order], v[order]
    starts = np.searchsorted(b, ok); ends = np.searchsorted(b, ok, side="right")
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        while True:
            perm = rng.permutation(len(ok))
            if not (perm == np.arange(len(ok))).any():
                break
        ys = np.empty_like(y)
        for k in range(len(ok)):
            d0, d1 = starts[perm[k]], ends[perm[k]]
            ys[starts[k]:ends[k]] = y[d0 + rng.integers(0, d1 - d0, ends[k] - starts[k])]
        out.append(auc_from(np.bincount(v[ys], minlength=256), np.bincount(v[~ys], minlength=256)))
    return out


def _elong(b, min_area, pad=False):
    # NOTE: exact eval_forme behaviour is pad=False: the distance transform runs on the letter's tight
    # bounding-box slice, so a component touching the slice edge gets an inflated inscribed radius. Kept
    # for comparability with the published 16-25 / ~11 numbers; pad=True is the border-corrected variant.
    if pad:
        b = np.pad(b, 1)
    lab, n = ndimage.label(b)
    if n == 0:
        return float("nan")
    d = ndimage.distance_transform_edt(b)
    a = np.bincount(lab.ravel())[1:].astype(float)
    r = np.asarray(ndimage.maximum(d, lab, range(1, n + 1)))
    g = a >= min_area
    if not g.any():
        return float("nan")
    e = a[g] / np.maximum(r[g], 1.0) ** 2
    o = np.argsort(e)
    return float(e[o][np.searchsorted(np.cumsum(a[g][o]), a[g].sum() / 2.0)])


def elongation(pred, label, support, pixel_um=9.612):
    """eval_forme.py (PR-32) generalised to pixel size. Returns dict(map=..., labels=..., letters=n)."""
    k = 9.612 / float(pixel_um)
    min_extent, min_area = 150.0 * k, 12.0 * k * k
    P = np.asarray(pred, np.float32)
    L = label.astype(bool) & support.astype(bool)
    lab, _ = ndimage.label(L)
    objs = [(i + 1, s) for i, s in enumerate(ndimage.find_objects(lab))
            if s is not None and max(s[0].stop - s[0].start, s[1].stop - s[1].start) >= min_extent]
    if not objs:
        return dict(map=float("nan"), labels=float("nan"), letters=0)
    t = np.quantile(np.concatenate([P[s][lab[s] == i] for i, s in objs]), 0.30)
    e = [_elong(P[s] >= t, min_area) for i, s in objs]
    le = [_elong(lab[s] == i, min_area) for i, s in objs]
    ep = [_elong(P[s] >= t, min_area, pad=True) for i, s in objs]
    return dict(map=float(np.nanmedian(e)), labels=float(np.nanmedian(le)), map_padded=float(np.nanmedian(ep)),
                letters=len(objs))


def full_report(pred, label, support, pixel_um, ci=True):
    r = dict(auc=auc(pred, label, support), sep=separation(pred, support),
             roll_null=label_roll_null(pred, label, support),
             ink_rate=float(label[support.astype(bool)].mean()) if support.any() else float("nan"))
    if ci:
        r["ci95"] = auc_ci(pred, label, support, n=300)
    r["elongation"] = elongation(pred, label, support, pixel_um)
    return r


def self_test():
    """Known-answer checks of this module (no scroll data)."""
    rng = np.random.default_rng(0)
    H, W = 512, 768
    label = np.zeros((H, W), bool)
    jit = rng.integers(-25, 26, (2, 4, 2))
    for r in range(2):  # non-overlapping letter-like strokes ("L" and "T" shapes, 6 px wide), jittered
        for c in range(4):
            y, x = 45 + r * 230 + jit[r, c, 0], 30 + c * 180 + jit[r, c, 1] + (60 if r else 0)
            if (r + c) % 2:
                label[y:y + 160, x:x + 6] = True; label[y + 154:y + 160, x:x + 90] = True     # L
            else:
                label[y:y + 6, x:x + 120] = True; label[y:y + 160, x + 57:x + 63] = True      # T
    support = np.ones_like(label)
    perfect = label.astype(np.float32)
    noisy = np.clip(0.5 * label + rng.normal(0.25, 0.2, label.shape), 0, 1)
    blob = ndimage.gaussian_filter(label.astype(np.float32), 12); blob /= blob.max()
    out = dict(auc_perfect=auc(perfect, label, support), auc_noisy=auc(noisy, label, support),
               auc_random=auc(rng.random(label.shape), label, support),
               roll_null_noisy=label_roll_null(noisy, label, support),
               elong_strokes=elongation(perfect, label, support)["map"],
               elong_blurred=elongation(blob, label, support)["map"])
    # rank-AUC cross-check on continuous scores (Mann-Whitney)
    from scipy.stats import mannwhitneyu
    q = to_u8(noisy)
    U = mannwhitneyu(q[label].astype(float), q[~label].astype(float)).statistic
    out["auc_noisy_mannwhitney"] = float(U / (label.sum() * (~label).sum()))
    checks = dict(perfect=out["auc_perfect"] == 1.0, random=abs(out["auc_random"] - 0.5) < 0.01,
                  mw_agrees=abs(out["auc_noisy"] - out["auc_noisy_mannwhitney"]) < 1e-9,
                  roll_null=abs(out["roll_null_noisy"] - 0.5) < 0.03,
                  strokes_more_elongated=out["elong_strokes"] > out["elong_blurred"])
    c2 = shuffle_floor(noisy, label, support, seeds=10)
    out["c2_floor_noisy_mean"] = float(np.mean(c2)); checks["c2_floor"] = abs(out["c2_floor_noisy_mean"] - 0.5) < 0.03
    # Known limitation (informational): on PERIODIC text (columns ~ W/2 apart) a half-width roll realigns
    # letters and the roll null rises above 0.5; the block-derangement C2 floor does not.
    per = np.zeros_like(label)
    for c in range(4):
        per[40:200, 30 + c * 192:36 + c * 192] = True; per[194:200, 30 + c * 192:120 + c * 192] = True
    pn = np.clip(0.5 * per + rng.normal(0.25, 0.2, per.shape), 0, 1)
    out["periodic_text_roll_null"] = label_roll_null(pn, per, support)
    out["periodic_text_c2_floor"] = float(np.mean(shuffle_floor(pn, per, support, seeds=10)))
    out["checks"] = checks; out["all_ok"] = all(checks.values())
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=1))
