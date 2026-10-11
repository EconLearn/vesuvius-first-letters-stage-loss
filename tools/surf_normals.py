"""SURF lane, Step 1a: local normals diagnostic at the organiser surfaces.

At organiser pixels q (geo.npz), compare three directions (sign-agnostic angles):
  n_fit : the fitted sheet normal at q (the code's own pulled-back radial covector, epsilon from cfg)
  n_las : the lasagna target normal at q (exactly as the loss decodes it: uint8 nx/ny at round(q/scale), nz=sqrt)
  n_org : the organiser mesh normal at q (geo.npz 'nrm')
and the dense_normals residual 1-|n.n_las| for n_fit (what the fit pays here) and for n_org (what a solution that
follows the organiser surface would pay here, if its normal equalled the organiser normal).
Reported by position along the sheet (column bins), because the measured winding drift is along the sheet.

usage: surf_normals.py CKPT UMB GEO_DIR[,..] LASAGNA_DIR OUT_JSON [--stride 8]
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import torch
import zarr

os.environ.setdefault('FIT_SPIRAL_TRITON', '0')
sys.path.insert(0, _P.SPIRAL_FIT_DIR)
sys.path.insert(0, f'{_P.WORK}/surf/scripts')
from surf_wind import build, winding_coord  # noqa: E402
from sample_spiral import get_radial_covector_in_scroll_space  # noqa: E402


def ang(a, b):
    c = np.abs((a * b).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def resid(a, b):
    c = np.abs((a * b).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return 1 - c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('umb'); ap.add_argument('geo'); ap.add_argument('las'); ap.add_argument('out')
    ap.add_argument('--stride', type=int, default=8); ap.add_argument('--device', default='cpu')
    ap.add_argument('--scale', type=int, default=4); ap.add_argument('--group', default='2')
    a = ap.parse_args()
    torch.set_num_threads(8)
    ck, cfg, model = build(a.ckpt, a.umb, a.device)
    dr = float(model.get_dr_per_winding().detach().cpu())
    eps = float(cfg.get('dense_normals_finite_difference_epsilon', 8.0))
    T = model.get_slice_to_spiral_transform()
    las = Path(a.las)
    nxz = zarr.open(str(next(las.glob('*_nx.ome.zarr')) / a.group), mode='r')
    nyz = zarr.open(str(next(las.glob('*_ny.ome.zarr')) / a.group), mode='r')
    res = {'ckpt': a.ckpt, 'iters': ck.get('completed_iterations'), 'eps': eps, 'stride': a.stride, 'windings': {}}
    for gd in a.geo.split(','):
        gd = Path(gd)
        g = np.load(gd / 'geo.npz')
        pts, nrm, band, lab = g['pts'], g['nrm'], g['band'].astype(bool), g['label'].astype(bool)
        sel = np.zeros_like(band); sel[::a.stride, ::a.stride] = band[::a.stride, ::a.stride]
        rows, cols = np.nonzero(sel)
        zyx = np.ascontiguousarray(pts[sel][:, ::-1]).astype(np.float32)
        norg = np.ascontiguousarray(nrm[sel][:, ::-1]).astype(np.float64)
        # fitted normal (the loss's own operator)
        nf, sp = [], []
        with torch.inference_mode():
            for s in range(0, len(zyx), 50000):
                q = torch.from_numpy(zyx[s:s + 50000]).to(a.device)
                spz = T(q)
                cov = get_radial_covector_in_scroll_space(T, q, spiral_zyx=spz, epsilon=eps)
                nf.append(cov.double().cpu().numpy()); sp.append(spz.float().cpu().numpy())
        nf = np.concatenate(nf); sp = np.concatenate(sp)
        v, th0, th, r = winding_coord(sp, dr)
        # lasagna target exactly as decoded by the loss
        idx = np.round(zyx / a.scale).astype(np.int64)
        zs = np.unique(idx[:, 0])
        nx8 = np.zeros(len(idx), np.float64); ny8 = np.zeros(len(idx), np.float64)
        # read per z-slab bounding box to keep IO small
        for z0 in range(zs.min(), zs.max() + 1, 32):
            m = (idx[:, 0] >= z0) & (idx[:, 0] < z0 + 32)
            if not m.any():
                continue
            ii = idx[m]
            y0, y1 = ii[:, 1].min(), ii[:, 1].max() + 1; x0, x1 = ii[:, 2].min(), ii[:, 2].max() + 1
            bx = np.asarray(nxz[z0:z0 + 32, y0:y1, x0:x1]); by = np.asarray(nyz[z0:z0 + 32, y0:y1, x0:x1])
            nx8[m] = bx[ii[:, 0] - z0, ii[:, 1] - y0, ii[:, 2] - x0]
            ny8[m] = by[ii[:, 0] - z0, ii[:, 1] - y0, ii[:, 2] - x0]
        valid = (nx8 != 0) | (ny8 != 0)
        nxd = (nx8 - 128.0) / 127.0; nyd = (ny8 - 128.0) / 127.0
        nzd = np.sqrt(np.clip(1 - nxd ** 2 - nyd ** 2, 0, None))
        nlas = np.stack([nzd, nyd, nxd], -1)
        nlas /= np.linalg.norm(nlas, axis=-1, keepdims=True) + 1e-12
        A_fo = ang(nf, norg); A_lo = ang(nlas, norg); A_fl = ang(nf, nlas)
        R_f = resid(nf, nlas); R_o = resid(norg, nlas)
        # fair comparison: organiser normals smoothed on the label grid (the fit is smooth at its 16-vox lattice)
        from scipy import ndimage
        smo = {}
        Hs, Ws = band.shape
        Nfull = np.zeros((Hs, Ws, 3)); Mfull = np.zeros((Hs, Ws))
        nn = nrm[..., ::-1].astype(np.float64)  # zyx
        sgn0 = np.sign(nn[..., 1:2] + 1e-9)  # consistent orientation (dominant y component)
        nn = nn * sgn0
        Mfull[band] = 1.0; Nfull[band] = nn[band]
        for sg in (8, 16, 32, 64):
            num = np.stack([ndimage.gaussian_filter(Nfull[..., c] * Mfull, sg) for c in range(3)], -1)
            den = ndimage.gaussian_filter(Mfull, sg)[..., None]
            ns = num / np.maximum(den, 1e-6); ns = ns[sel]
            ns /= np.linalg.norm(ns, axis=-1, keepdims=True) + 1e-12
            smo[sg] = ns
        R_os = {sg: resid(smo[sg], nlas) for sg in smo}
        A_los = {sg: ang(smo[sg], nlas) for sg in smo}
        # tilt decomposition of fit vs organiser: component of the difference along z (out-of-slice) vs in-slice
        def tilt_z(n):
            n = n / np.linalg.norm(n, axis=-1, keepdims=True)
            return np.degrees(np.arcsin(np.clip(n[:, 0], -1, 1)))  # elevation of the normal out of the slice plane
        def inplane_dir(n):
            return np.degrees(np.arctan2(n[:, 1], n[:, 2]))
        dz_fo = tilt_z(nf * np.sign((nf * norg).sum(-1, keepdims=True))) - tilt_z(norg)
        dip = inplane_dir(nf * np.sign((nf * norg).sum(-1, keepdims=True))) - inplane_dir(norg)
        dip = (dip + 180) % 360 - 180
        dz_lo = tilt_z(nlas * np.sign((nlas * norg).sum(-1, keepdims=True))) - tilt_z(norg)
        dip_lo = inplane_dir(nlas * np.sign((nlas * norg).sum(-1, keepdims=True))) - inplane_dir(norg)
        dip_lo = (dip_lo + 180) % 360 - 180
        W = band.shape[1]
        bins = np.linspace(cols.min(), cols.max() + 1, 9).astype(int)
        per = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (cols >= b0) & (cols < b1) & valid
            if m.sum() < 20:
                continue
            per.append({'cols': [int(b0), int(b1)], 'n': int(m.sum()),
                        'v_med': float(np.median(v[m])),
                        'ang_fit_org_med': float(np.median(A_fo[m])), 'ang_las_org_med': float(np.median(A_lo[m])),
                        'ang_fit_las_med': float(np.median(A_fl[m])),
                        'resid_fit_mean': float(R_f[m].mean()), 'resid_org_mean': float(R_o[m].mean()),
                        'inplane_fit_minus_org_med': float(np.median(dip[m])),
                        'inplane_las_minus_org_med': float(np.median(dip_lo[m])),
                        'elev_fit_minus_org_med': float(np.median(dz_fo[m])),
                        'elev_las_minus_org_med': float(np.median(dz_lo[m]))})
        m = valid
        summ = {'n': int(m.sum()), 'valid_frac': float(valid.mean()),
                'ang_fit_org_med': float(np.median(A_fo[m])), 'ang_las_org_med': float(np.median(A_lo[m])),
                'ang_fit_las_med': float(np.median(A_fl[m])),
                'resid_fit_mean': float(R_f[m].mean()), 'resid_org_mean': float(R_o[m].mean()),
                'resid_fit_minus_org': float((R_f[m] - R_o[m]).mean()),
                'resid_org_smoothed_mean': {int(sg): float(R_os[sg][m].mean()) for sg in R_os},
                'robust_trunc': {f'{deg}deg': {'fit': float(np.minimum(R_f[m], 1 - np.cos(np.radians(deg))).mean()),
                                               'org_s16': float(np.minimum(R_os[16][m], 1 - np.cos(np.radians(deg))).mean()),
                                               'org_raw': float(np.minimum(R_o[m], 1 - np.cos(np.radians(deg))).mean())}
                                 for deg in (5, 10, 20, 45)},
                'resid_quantiles_fit': [float(np.quantile(R_f[m], q)) for q in (0.5, 0.9, 0.99)],
                'resid_quantiles_org_s16': [float(np.quantile(R_os[16][m], q)) for q in (0.5, 0.9, 0.99)],
                'ang_las_orgsmoothed_med': {int(sg): float(np.median(A_los[sg][m])) for sg in A_los},
                'inplane_fit_minus_org_med': float(np.median(dip[m])), 'inplane_las_minus_org_med': float(np.median(dip_lo[m])),
                'elev_fit_minus_org_med': float(np.median(dz_fo[m])), 'elev_las_minus_org_med': float(np.median(dz_lo[m]))}
        res['windings'][gd.name] = {'summary': summ, 'by_column': per}
        print(gd.name, json.dumps(summ), flush=True)
        for p in per:
            print('  ', json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in p.items()}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
