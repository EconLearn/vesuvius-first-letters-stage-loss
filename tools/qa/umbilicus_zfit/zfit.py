"""Reference implementation of the --zfit robust local-linear fit (LOWESS: tricube window of half-width H in z, bisquare
robustness on the 2D residual norm over 6 x its median, 3 iterations). The C++ in vc_gen_umbilicus must match this.
pts: array (n, 3) of (z, x, y) estimates (several per z allowed). zq: slices to evaluate. Returns (len(zq), 2)."""
import numpy as np


def tricube(u):
    u = np.clip(np.abs(u), 0, 1)
    return (1 - u ** 3) ** 3


def local_linear(z, xy, w, z0):
    sw = w.sum()
    if np.count_nonzero(w > 0) < 3 or sw <= 0:
        return None
    zm = (w * z).sum() / sw
    dz = z - zm
    szz = (w * dz * dz).sum()
    m = (w[:, None] * xy).sum(0) / sw
    if szz <= 1e-12 * max(1.0, sw):
        return m
    slope = (w[:, None] * dz[:, None] * (xy - m)).sum(0) / szz
    return m + slope * (z0 - zm)


def zfit(pts, zq, H, iters=3):
    pts = np.asarray(pts, float)
    z, xy = pts[:, 0], pts[:, 1:3]
    rob = np.ones(len(z))
    for it in range(iters + 1):
        # fitted value at every estimate (for the residuals), then at the query slices on the last pass
        if it < iters:
            fit = np.empty_like(xy)
            for i in range(len(z)):
                v = local_linear(z, xy, tricube((z - z[i]) / H) * rob, z[i])
                fit[i] = xy[i] if v is None else v
            r = np.linalg.norm(xy - fit, axis=1)
            s = np.median(r)
            if s <= 0:
                break
            u = np.clip(r / (6 * s), 0, 1)
            rob = (1 - u ** 2) ** 2
    out = np.empty((len(zq), 2))
    for k, z0 in enumerate(zq):
        v = local_linear(z, xy, tricube((z - z0) / H) * rob, z0)
        if v is None:
            sel = z == z0
            v = np.median(xy[sel], 0) if sel.any() else np.full(2, np.nan)
        out[k] = v
    return out
