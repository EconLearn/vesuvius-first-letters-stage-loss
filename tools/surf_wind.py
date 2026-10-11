"""SURF lane: checkpoint-native winding-consistency metric.

For a fitted spiral checkpoint, map every organiser-surface pixel q (geo.npz: XYZ positions on the label grid) through
the fit's own slice->spiral transform and read its continuous winding coordinate
    v(q) = r/dr - theta'/2pi - theta0/2pi,  theta' = (theta - theta0) mod 2pi
(theta0 = seam placed opposite the organiser block, so v is continuous over the block). v is an integer exactly on
fitted windings. n(q) = round(v) = the fitted winding nearest to q (what a flattened-grid nearest correspondence picks).
A fit that keeps the organiser sheet on ONE winding has n constant over the block.

Metrics (per organiser winding, over text px = label & band, and over all band px):
  share_mode      : max_k share(n == k)      (analogue of G3's densest-200-column share)
  n_ge1pct        : number of fitted windings holding >= 1% of the px
  jump_x8 / jump_z8 : fraction of 8-px neighbour pairs (along the line / along z) with different n
  sw_per_1000px   : switches per 1,000 px of path = mean(jump_x8, jump_z8) * 1000 / 8
  v_p5_p95        : spread of v
  dvdz_per_1000   : least-squares slope of v vs z (windings per 1,000 slices) -> the drift that tilt produces
  frac_abs_lt_0.25: share of px whose |v - round(v)| < 0.25 (sheet sits near a fitted winding)
Robust co-primaries (added 2026-10-09 after a synthetic test showed n = round(v) switches explode when v sits near a
half-integer: a hop-free sheet at phase 0.5 scores share 0.50 and 61.8 switches per 1,000 px):
  best_phase_share       : max over offsets c of share(floor(v - c) == mode), i.e. text inside one winding-wide window
  sw_smooth32_per_1000px : switches after Gaussian smoothing of v (sigma 32 px)
  col_median_drift_range : range of per-200-px-column median v along the sheet (drift = real hopping)
  between_block_dv_w060_minus_w058 : fitted windings between the two organiser blocks (truth on PHerc1447: 2.0)

usage: surf_wind.py CKPT UMBILICUS_JSON GEO_DIR[,GEO_DIR...] OUT_JSON [--stride 2] [--device cpu]
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import sys, os, json, argparse, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault('FIT_SPIRAL_TRITON', '0')
SF = _P.SPIRAL_FIT_DIR
sys.path.insert(0, SF)
from checkpoint_io import load_checkpoint_cpu  # noqa: E402
import flatten_spiral_checkpoint as FSC  # noqa: E402


def build(ckpt_path, umb, device):
    ck = load_checkpoint_cpu(str(ckpt_path))
    cfg = FSC._checkpoint_config(ck)
    model = FSC._build_model(ck, cfg, Path(umb), torch.device(device))
    return ck, cfg, model


@torch.inference_mode()
def spiral_coords(model, zyx, device, chunk=200_000):
    T = model.get_slice_to_spiral_transform()
    out = []
    for s in range(0, zyx.shape[0], chunk):
        q = torch.from_numpy(zyx[s:s + chunk]).to(device)
        out.append(T(q).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def winding_coord(sp, dr, theta0=None):
    y, x = sp[:, 1].astype(np.float64), sp[:, 2].astype(np.float64)
    th = np.mod(np.arctan2(y, x), 2 * np.pi)
    r = np.hypot(y, x)
    if theta0 is None:
        c = np.angle(np.mean(np.exp(1j * th)))  # circular mean of the block
        theta0 = np.mod(c + np.pi, 2 * np.pi)
    thp = np.mod(th - theta0, 2 * np.pi)
    v = r / dr - thp / (2 * np.pi) - theta0 / (2 * np.pi)
    return v, theta0, th, r


STRIDE = [2]
SAVE_MAPS = True


def metrics(n, v, z, grid_shape, mask, sel, label):
    """n, v, z: per selected grid pixel arrays aligned with np.nonzero(sel)."""
    out = {}
    m = mask
    if m.sum() == 0:
        return {'px': 0}
    nm = n[m]
    ks, cs = np.unique(nm, return_counts=True)
    share = cs / cs.sum()
    out['px'] = int(m.sum())
    out['share_mode'] = float(share.max())
    out['mode_winding'] = int(ks[np.argmax(cs)])
    out['n_ge1pct'] = int((share >= 0.01).sum())
    out['windings_hist'] = {int(k): round(float(s), 4) for k, s in zip(ks, share) if s >= 0.005}
    vm = v[m]
    out['v_p5_p50_p95'] = [float(np.percentile(vm, q)) for q in (5, 50, 95)]
    fr = np.abs(vm - np.round(vm))
    out['frac_abs_lt_0.25'] = float((fr < 0.25).mean())
    zm = z[m]
    A = np.stack([zm - zm.mean(), np.ones_like(zm)], 1)
    coef = np.linalg.lstsq(A, vm - vm.mean(), rcond=None)[0]
    out['dvdz_per_1000'] = float(coef[0] * 1000)
    # neighbour jumps on the label grid (8 px apart), pairs where both pixels are in mask
    G = np.full(grid_shape, np.iinfo(np.int32).min, np.int32)
    idx = np.nonzero(sel)
    G[idx] = n.astype(np.int32)
    M = np.zeros(grid_shape, bool)
    M[idx] = mask
    k = max(1, 8 // STRIDE[0])
    jx = (M[:, :-k] & M[:, k:]); jz = (M[:-k, :] & M[k:, :])
    dx = (G[:, :-k] != G[:, k:])[jx]; dz = (G[:-k, :] != G[k:, :])[jz]
    out['pair_px'] = k * STRIDE[0]
    out['jump_x8'] = float(dx.mean()) if dx.size else None
    out['jump_z8'] = float(dz.mean()) if dz.size else None
    if dx.size and dz.size:
        out['sw_per_1000px'] = float((out['jump_x8'] + out['jump_z8']) / 2 * 1000 / (k * STRIDE[0]))
    # ---- robust co-primary metrics (added 2026-10-09, post hoc; see FINDINGS.md section 1.4) ----
    from scipy import ndimage as _ndi
    V = np.full(grid_shape, np.nan); V[idx] = v
    Mg = np.zeros(grid_shape, bool); Mg[idx] = mask
    best = 0.0; bestc = 0.0
    for c in np.arange(0, 1, 0.05):
        vals, cnt = np.unique(np.floor(V[Mg] - c).astype(np.int64), return_counts=True)
        if cnt.max() / cnt.sum() > best:
            best, bestc = float(cnt.max() / cnt.sum()), float(c)
    out['best_phase_share'] = best; out['best_phase_c'] = bestc
    sg = 32.0 / STRIDE[0]
    A = _ndi.gaussian_filter(np.where(Mg, V, 0.0), sg); Wt = _ndi.gaussian_filter(Mg.astype(float), sg)
    Vs = np.where(Wt > 1e-3, A / np.maximum(Wt, 1e-9), np.nan)
    Ns = np.round(np.nan_to_num(Vs, nan=-999)).astype(np.int64)
    sx = (Ns[:, :-k] != Ns[:, k:])[jx]; sz = (Ns[:-k, :] != Ns[k:, :])[jz]
    if sx.size and sz.size:
        out['sw_smooth32_per_1000px'] = float((sx.mean() + sz.mean()) / 2 * 1000 / (k * STRIDE[0]))
    out['jitter_std_s32'] = float(np.nanstd((V - Vs)[Mg]))
    blk = max(1, int(200 / STRIDE[0])); cols = []
    for c0 in range(0, grid_shape[1], blk):
        mm = Mg[:, c0:c0 + blk]
        if mm.sum() > 50:
            cols.append(float(np.median(V[:, c0:c0 + blk][mm])))
    out['col_median_v'] = [round(x, 2) for x in cols]
    out['col_median_drift_range'] = float(max(cols) - min(cols)) if cols else None
    out['v_median'] = float(np.median(v[m]))
    return out


def run(ckpt, umb, geo_dirs, out_json, stride=2, device='cpu', tag=None):
    t0 = time.time()
    STRIDE[0] = stride
    ck, cfg, model = build(ckpt, umb, device)
    dr = float(model.get_dr_per_winding().detach().cpu())
    res = {'ckpt': str(ckpt), 'completed_iterations': ck.get('completed_iterations'), 'dr_per_winding': dr,
           'z': [ck['z_begin'], ck['z_end']], 'stride': stride, 'tag': tag, 'windings': {}}
    for gd in geo_dirs:
        gd = Path(gd)
        g = np.load(gd / 'geo.npz')
        pts, band = g['pts'], g['band'].astype(bool)
        lab = g['label'].astype(bool) if 'label' in g.files else np.zeros_like(band)
        H, W = band.shape
        sel = np.zeros_like(band)
        sel[::stride, ::stride] = band[::stride, ::stride]
        xyz = pts[sel]
        zyx = np.ascontiguousarray(xyz[:, ::-1]).astype(np.float32)
        # keep only points inside the fit's z range
        sp = spiral_coords(model, zyx, device)
        v, th0, th, r = winding_coord(sp, dr)
        n = np.round(v).astype(np.int64)
        sub = (slice(None, None, stride), slice(None, None, stride))
        gshape = band[sub].shape
        sel_s = sel[sub]
        lab_s = (lab & band)[sub][sel_s]
        all_s = np.ones(int(sel_s.sum()), bool)
        z = zyx[:, 0]
        r_text = metrics(n, v, z, gshape, lab_s, sel_s, 'text')
        r_band = metrics(n, v, z, gshape, all_s, sel_s, 'band')
        # stride-corrected switch rate: neighbours are 8 grid px * stride apart in the original grid
        if SAVE_MAPS:
            V = np.full(gshape, np.nan, np.float32); V[np.nonzero(sel_s)] = v
            np.savez_compressed(Path(out_json).with_suffix('') .as_posix() + f'_{gd.name}_v.npz', v=V, lab=(lab & band)[sub], z=pts[sub][..., 2], x=pts[sub][..., 0], y=pts[sub][..., 1])
        res['windings'][gd.name] = {'theta0': float(th0), 'text': r_text, 'band': r_band,
                                    'r_spiral_median': float(np.median(r))}
        print(gd.name, json.dumps({'text': {k: r_text.get(k) for k in ('px', 'share_mode', 'n_ge1pct', 'sw_per_1000px', 'dvdz_per_1000', 'v_p5_p50_p95')},
                                   'band': {k: r_band.get(k) for k in ('px', 'share_mode', 'n_ge1pct', 'sw_per_1000px', 'dvdz_per_1000')}}), flush=True)
    ws = res['windings']
    if 'w058' in ws and 'w060' in ws and ws['w058']['text'].get('v_median') is not None and ws['w060']['text'].get('v_median') is not None:
        res['between_block_dv_w060_minus_w058'] = ws['w060']['text']['v_median'] - ws['w058']['text']['v_median']  # truth: 2.0
    res['seconds'] = time.time() - t0
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('umb'); ap.add_argument('geo'); ap.add_argument('out')
    ap.add_argument('--stride', type=int, default=2); ap.add_argument('--device', default='cpu')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    run(a.ckpt, a.umb, a.geo.split(','), a.out, a.stride, a.device, a.tag)
