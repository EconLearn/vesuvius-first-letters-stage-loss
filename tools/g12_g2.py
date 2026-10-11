#!/usr/bin/env python
"""G2 stage-loss decomposition, PHerc1447 w058/w060, scoring band z in [ZLO, ZHI) (fit band 13100-14100).

Arms (all scored on the SAME organiser label pixels: certainty mask ∩ band):
  A     organiser surface + organiser 24-layer render + ink_9um  (E1 arm A predictions, = the G0-type number here)
  R     organiser surface + OUR 28-layer render (validated renderer) + ink_9um, scored directly
  Rm    R mapped through the 3D nearest-surface correspondence (instrument identity check: must equal R, d ~ 0)
  P10 / M10   SABOTAGE: organiser surface shifted +10 / -10 voxels along its normal, rendered, inferred, mapped back
  X1    SABOTAGE: the adjacent organiser winding (w060 for w058 and vice versa) as the surface (real +-1 sheet)
  auto_<tag>  the automatic spiral-fit (+ flatten) surface, rendered 28 layers (flip-normals per catalogue), mapped back
Mapping: for each organiser label pixel (3D point on the organiser surface), nearest rendered pixel (offset-0 3D point)
of the arm's surface (cKDTree); prediction transferred; distance = adherence / registration error.
End-to-end AUC: uncovered pixels (d > RCOV) get prediction 0. Covered AUC: only d <= RCOV.

usage: g12_g2.py org [--procs N] | auto TIFXYZ TAG [--procs N] | infer NAME... | score [TAG...]
"""
from __future__ import annotations
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import argparse, json, sys, time
from pathlib import Path
import numpy as np, tifffile, zarr
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, _P.TOOLS)
import g12lib as L  # noqa: E402
from vlib import metrics as M, guards  # noqa: E402

W = Path(f'{_P.WORK}/g1g2'); OUT = Path(f'{_P.OUT}/g1g2'); ARMS = W / 'arms'
E1 = Path(f'{_P.WORK}/e1'); HF = Path(f'{_P.DATA}/hf/ink-8um-pherc1447-surfaces')
ZLO, ZHI = 13150, 14050
NS = 28
CKPT = f'{_P.CKPT}/ink_9um/hybrid_3d2d-seed42/step-075000.pth'
PIX = 8.64
WINDS = ['w058', 'w060']
RCOV = 14.0      # primary coverage cap = half the measured w058-w060 sheet pitch (~28 vox): beyond it the nearest
                 # automatic sheet is closer to the neighbouring winding than to the labelled one
RCOV_STRICT = 6.0
DOSE = [('P2', 2.0), ('M2', -2.0), ('P4', 4.0), ('M4', -4.0), ('P6', 6.0), ('M6', -6.0)]   # offset dose-response
PHASES = [(16, 16), (32, 32), (48, 48), (32, 0)]   # tiling-phase controls (zero-pad top/left, crop back)
RMAX = 60.0      # KD query cap


def write_zarr(path, vol):
    g = zarr.open_group(str(path), mode='w', zarr_format=2)
    a = g.create_array('0', shape=vol.shape, chunks=(vol.shape[0], 128, 128), dtype='uint8')
    a[:] = vol


# ------------------------------------------------------------------------------------------------ organiser arms
def stage_org(procs):
    m = L.ChunkMirror(); ARMS.mkdir(parents=True, exist_ok=True)
    for w in WINDS:
        d = W / 'org' / w; d.mkdir(parents=True, exist_ok=True)
        if not (d / 'geo.npz').exists():
            prep = json.loads((E1 / f'1447_{w}' / 'prep.json').read_text())
            i0, i1, j0, j1 = prep['meta']['crop']
            z = np.load(E1 / f'1447_{w}' / 'labels.npz'); lab, sup = z['label'], z['support']
            P, valid, meta = L.load_tifxyz(HF / w / 'tifxyz'); N = L.vertex_normals(P)
            pts, nrm = L.surface_at_pixels(P, N, np.arange(i0, i1), np.arange(j0, j1), meta['scale'][0])
            zc = pts[..., 2]
            band = sup & np.isfinite(zc) & (zc >= ZLO) & (zc < ZHI) & np.isfinite(nrm).all(-1)
            rr = np.nonzero(band.any(1))[0]; cc = np.nonzero(band.any(0))[0]
            r0, r1 = max(0, rr.min() - 160), min(lab.shape[0], rr.max() + 161)
            c0, c1 = max(0, cc.min() - 160), min(lab.shape[1], cc.max() + 161)
            s = (slice(r0, r1), slice(c0, c1))
            np.savez(d / 'geo.npz', pts=pts[s], nrm=nrm[s], label=lab[s], band=band[s], sub=[r0, r1, c0, c1],
                     e1crop=[i0, i1, j0, j1])
            print(w, 'band px', int(band.sum()), 'ink px', int((lab & band).sum()), 'sub', [r0, r1, c0, c1], flush=True)
        g = np.load(d / 'geo.npz')
        for arm, eo in [('R', 0.0), ('P10', 10.0), ('M10', -10.0)] + DOSE:
            zp = ARMS / f'{w}_{arm}.zarr'
            if zp.exists():
                continue
            t = time.time()
            vol, v = L.render_pixels(m, g['pts'], g['nrm'], NS, 1.0, False, extra_offset=eo, procs=procs)
            write_zarr(zp, vol)
            print(w, arm, vol.shape, f'{time.time() - t:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ automatic surface
def stage_auto(tifxyz, tag, procs, R=30.0, margin_px=96, flip=True, mode='nearest'):
    m = L.ChunkMirror(); ARMS.mkdir(parents=True, exist_ok=True)
    P, valid, meta = L.load_tifxyz(tifxyz); N = L.vertex_normals(P); s = float(meta['scale'][0])
    Q = np.concatenate([np.load(W / 'org' / w / 'geo.npz')['pts'][np.load(W / 'org' / w / 'geo.npz')['band']] for w in WINDS])
    near = np.zeros(valid.shape, bool)
    if mode == 'nearest':
        # only grid vertices that are the nearest automatic vertex (<= RMAX) of some organiser label point: exactly
        # what the 3D correspondence can use; dilated below for the mapping's sub-cell search and inference context
        vi = np.argwhere(valid)
        Tg = cKDTree(P[valid])
        d, j = Tg.query(Q[::3], distance_upper_bound=RMAX, workers=4)
        j = np.unique(j[np.isfinite(d)])
        near[vi[j, 0], vi[j, 1]] = True
        dil = 4
    else:
        T = cKDTree(Q[::5])
        d, _ = T.query(P[valid], distance_upper_bound=R + 5)
        near[valid] = d <= R
        dil = max(1, int(round(margin_px * s)))
    print(tag, 'grid', P.shape, 'scale', s, 'valid', int(valid.sum()), 'near-band vertices', int(near.sum()), 'mode', mode, flush=True)
    keep = ndimage.binary_dilation(near, iterations=dil) & valid
    lab, n = ndimage.label(keep, structure=np.ones((3, 3)))
    recs = []
    for k, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None or (near[sl] & (lab[sl] == k + 1)).sum() < 5:
            continue
        rows = np.arange(int(np.floor(sl[0].start / s)), int(np.ceil(sl[0].stop / s)))
        cols = np.arange(int(np.floor(sl[1].start / s)), int(np.ceil(sl[1].stop / s)))
        pts, nrm = L.surface_at_pixels(P, N, rows, cols, s)
        # blank pixels whose nearest grid vertex is outside the kept component (limits render cost)
        gr = np.clip(np.rint((rows + 0.5) * s).astype(int), 0, P.shape[0] - 1)
        gc = np.clip(np.rint((cols + 0.5) * s).astype(int), 0, P.shape[1] - 1)
        kmask = (lab[np.ix_(gr, gc)] == k + 1)
        pts[~kmask] = np.nan
        name = f'auto_{tag}_c{k}'
        t = time.time()
        vol, v = L.render_pixels(m, pts, nrm, NS, 1.0, flip, procs=procs)
        write_zarr(ARMS / f'{name}.zarr', vol)
        np.save(ARMS / f'{name}_pts.npy', pts.astype(np.float32))
        recs.append(dict(name=name, grid_rows=[sl[0].start, sl[0].stop], grid_cols=[sl[1].start, sl[1].stop],
                         shape=list(vol.shape), valid_px=int(v.sum()), secs=round(time.time() - t, 1)))
        print(json.dumps(recs[-1]), flush=True)
    (ARMS / f'auto_{tag}.json').write_text(json.dumps(dict(tifxyz=str(tifxyz), scale=s, flip=flip, R=R, mode=mode, comps=recs), indent=1))


# ------------------------------------------------------------------------------------------------ inference
def stage_asub():
    """Control for crop-dependent inference: the organiser's own 24 layers (E1 arm A input) cropped to the G2 sub-crop."""
    for w in WINDS:
        g = np.load(W / 'org' / w / 'geo.npz'); r0, r1, c0, c1 = g['sub']
        a = zarr.open(str(E1 / f'1447_{w}' / 'A.zarr'), mode='r')['0']
        write_zarr(ARMS / f'{w}_Asub.zarr', np.asarray(a[:, r0:r1, c0:c1]))
        print(w, 'Asub', a.shape, '->', (a.shape[0], r1 - r0, c1 - c0), flush=True)


def stage_phase():
    """Tiling-phase noise floor: the R arm re-inferred with the canvas zero-padded by (dr, dc) at top/left."""
    for w in WINDS:
        a = np.asarray(zarr.open(str(ARMS / f'{w}_R.zarr'), mode='r')['0'])
        for dr, dc in PHASES:
            zp = ARMS / f'{w}_Rph{dr}_{dc}.zarr'
            if not zp.exists():
                write_zarr(zp, np.pad(a, ((0, 0), (dr, 0), (dc, 0))))
                print(w, 'phase', dr, dc, flush=True)


def stage_ofs(tag, procs):
    """Organiser layout + the automatic surface's measured offset field (decomposition arms).
    OFSn_<tag>: pts + s(q) n_org(q) rendered along the organiser normals   (offset waviness only)
    OFSt_<tag>: same displaced points, rendered along the displaced surface's own normals (waviness + tilt)
    s(q) = signed offset of the nearest automatic point along n_org, |d| <= RCOV, median-filtered 5 px; else 0."""
    m = L.ChunkMirror()
    meta = json.loads((ARMS / f'auto_{tag}.json').read_text())
    P = np.concatenate([np.load(ARMS / f'{c["name"]}_pts.npy').reshape(-1, 3) for c in meta['comps']])
    P = P[np.isfinite(P).all(1)]
    T = cKDTree(P, balanced_tree=False, compact_nodes=False)
    for w in WINDS:
        g = np.load(W / 'org' / w / 'geo.npz'); pts, nrm = g['pts'], g['nrm']
        ok = np.isfinite(pts).all(-1) & np.isfinite(nrm).all(-1)
        d, j = T.query(pts[ok], distance_upper_bound=RCOV, workers=4)
        hit = np.isfinite(d); jf = np.where(hit, j, 0)
        sv = np.where(hit, ((P[jf] - pts[ok]) * nrm[ok]).sum(1), 0.0)
        S = np.zeros(ok.shape, np.float32); S[ok] = sv
        S = ndimage.median_filter(S, size=5)
        disp = pts + S[..., None] * nrm
        dn = np.cross(np.gradient(disp, axis=1), np.gradient(disp, axis=0))
        dn = dn / np.linalg.norm(dn, axis=-1, keepdims=True)
        dn = np.where(((dn * nrm).sum(-1) < 0)[..., None], -dn, dn)        # keep the organiser's normal sign
        np.save(ARMS / f'{w}_OFS_{tag}_S.npy', S)
        for arm, nn in [(f'OFSn_{tag}', nrm), (f'OFSt_{tag}', dn.astype(np.float32))]:
            zp = ARMS / f'{w}_{arm}.zarr'
            if zp.exists():
                continue
            t = time.time()
            vol, v = L.render_pixels(m, disp.astype(np.float32), nn, NS, 1.0, False, procs=procs)
            write_zarr(zp, vol)
            print(w, arm, vol.shape, f'{time.time() - t:.0f}s', 'offset field |S| median %.2f' % float(np.median(np.abs(S[g['band']]))), flush=True)


def stage_infer_cpu(names, threads=8, workers=3):
    """Same official villa flat inference, on CPU in fp32 (autocast off). Output: ARMS/<name>c_pred[_reverse].tif.
    Used when the shared GPU is saturated by other lanes; compare only against other CPU-run arms (paired device)."""
    import logging, multiprocessing as mp
    from dataclasses import replace
    mp.set_start_method('fork', force=True)
    import torch
    torch.set_num_threads(threads)
    from vesuvius.ink_detection.inference import infer as INF
    logging.basicConfig(level=logging.WARNING)
    todo = [n for n in names if not (ARMS / f'{n}c_pred_reverse.tif').exists()]
    if not todo:
        print('nothing to infer'); return
    args = INF.parse_args([str(ARMS / f'{todo[0]}.zarr'), CKPT, str(ARMS / 'x.tif'), '--direction', 'both',
                           '--batch-size', '16', '--num-workers', str(workers), '--no-compile', '--amp-dtype', 'default'])
    args = INF.normalize_inference_paths(args)
    t = time.time()
    configured = INF.configure_model(args)
    device = torch.device('cpu')
    configured = replace(configured, model=configured.model.to(device).eval())
    print(f'cpu model ready {time.time() - t:.0f}s amp={configured.amp_dtype}', flush=True)
    for n in todo:
        t = time.time(); out = ARMS / f'{n}c_pred.tif'
        for direction in ('forward', 'reverse'):
            INF.infer_single_zarr(args=args, input_zarr=str(ARMS / f'{n}.zarr'), configured_model=configured, device=device,
                                  output_tiff=INF.resolve_single_output_path(out, direction=direction, requested_direction='both'),
                                  layer_direction=direction)
        print(f'infercpu {n} {time.time() - t:.0f}s', flush=True)


def stage_infer(names, workers=3):
    import logging, multiprocessing as mp, resource
    # villa flat inference leaks pipe fds per canvas with fork-started DataLoader workers (py3.14): the it10000 run
    # hit EMFILE after 49 canvases at the default soft limit 1024 (measured). Raise the soft limit to the hard limit.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
    from dataclasses import replace
    mp.set_start_method('fork', force=True)
    from vesuvius.ink_detection.inference import infer as INF
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] %(levelname)s %(message)s')
    import torch
    torch.set_float32_matmul_precision('high')
    todo = [n for n in names if not (ARMS / f'{n}_pred_reverse.tif').exists()]
    if not todo:
        print('nothing to infer'); return
    args = INF.parse_args([str(ARMS / f'{todo[0]}.zarr'), CKPT, str(ARMS / 'x.tif'), '--direction', 'both',
                           '--batch-size', '32', '--num-workers', str(workers), '--no-compile'])
    args = INF.normalize_inference_paths(args)
    guards.gpu_check(1800, wait_s=6 * 3600)
    t = time.time()
    configured = INF.configure_model(args)
    model, device = INF.prepare_model_for_inference(configured.model, gpu_ids=args.gpu_ids, compile_model=False,
                                                    compile_mode=args.compile_mode)
    configured = replace(configured, model=model)
    print(f'model ready {time.time() - t:.0f}s amp={configured.amp_dtype} overlap/blend per CLI defaults', flush=True)
    for n in todo:
        t = time.time(); out = ARMS / f'{n}_pred.tif'
        # no per-canvas memory gate: this warm process already holds its CUDA cache (peak 1.32 GB, measured), so a
        # further canvas needs no new memory; the gate before model load (1.8 GB free) protects the shared card
        for direction in ('forward', 'reverse'):
            INF.infer_single_zarr(args=args, input_zarr=str(ARMS / f'{n}.zarr'), configured_model=configured, device=device,
                                  output_tiff=INF.resolve_single_output_path(out, direction=direction, requested_direction='both'),
                                  layer_direction=direction)
        print(f'infer {n} {time.time() - t:.0f}s peak_alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB', flush=True)


# ------------------------------------------------------------------------------------------------ scoring
def _pred(path):
    p = tifffile.imread(path).astype(np.float32)
    return p / (255.0 if p.max() > 1.0 else 1.0)


def load_pair(name):
    return _pred(ARMS / f'{name}_pred.tif'), _pred(ARMS / f'{name}_pred_reverse.tif')


def _shift(name):
    import re
    m = re.fullmatch(r'([PM])(\d+)', name)
    return None if not m else (1 if m.group(1) == 'P' else -1) * float(m.group(2))


def _canvas_normals(pts):
    """unit normals of a rendered canvas from finite differences of its 3D points (sign arbitrary)."""
    dr = np.gradient(pts, axis=0); dc = np.gradient(pts, axis=1)
    n = np.cross(dc, dr)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def surface_arm(name, w):
    """(positions (n,3), pred_fwd, pred_rev, canvas index or None, unit normals (n,3)) for one arm (offset-0 points)."""
    sh = 0.0 if name in ('R', 'X1') else _shift(name)
    if sh is not None:
        ww = w if name != 'X1' else [x for x in WINDS if x != w][0]
        g = np.load(W / 'org' / ww / 'geo.npz')
        pos = g['pts'] + sh * g['nrm']
        pf, pr = load_pair(f'{ww}_{"R" if name == "X1" else name}')
        ok = np.isfinite(pos).all(-1)
        return pos[ok], pf[ok], pr[ok], np.argwhere(ok), g['nrm'][ok]
    meta = json.loads((ARMS / f'{name}.json').read_text())
    P, F, Rv, Nn = [], [], [], []
    for c in meta['comps']:
        pos = np.load(ARMS / f'{c["name"]}_pts.npy'); pf, pr = load_pair(c['name'])
        nn = _canvas_normals(pos)
        ok = np.isfinite(pos).all(-1) & np.isfinite(nn).all(-1)
        P.append(pos[ok]); F.append(pf[ok]); Rv.append(pr[ok]); Nn.append(nn[ok])
    return np.concatenate(P), np.concatenate(F), np.concatenate(Rv), None, np.concatenate(Nn)


def block_pair(pa, pb, label, sup):
    """block histograms of two prediction canvases on the same support (for paired bootstrap)."""
    return M.block_hists(M.to_u8(pa), label, sup), M.block_hists(M.to_u8(pb), label, sup)


def paired_ci(ha, hb, n=1000, seed=0):
    rng = np.random.default_rng(seed); k = len(ha[0]); out = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        out.append(M.auc_from(hb[0][i].sum(0), hb[1][i].sum(0)) - M.auc_from(ha[0][i].sum(0), ha[1][i].sum(0)))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def score_canvas(pf, pr, label, sup, c2=True, order=None):
    """order: fixed a-priori depth order ('reverse' for organiser renders per the published README; 'forward' for the
    automatic renders made with flip-normals, equivalent by measured geometry). None -> label-free separation rule."""
    sf, sr = M.separation(pf, sup), M.separation(pr, sup)
    ch_sep = M.choose_direction(sf, sr) or 'forward'
    ch = order or ch_sep
    p = pf if ch == 'forward' else pr
    r = dict(auc_fwd=M.auc(pf, label, sup), auc_rev=M.auc(pr, label, sup), chosen=ch, chosen_sep=ch_sep, sep=[sf, sr],
             support_px=int(sup.sum()), ink_px=int((label & sup).sum()))
    r['auc'] = r['auc_fwd'] if ch == 'forward' else r['auc_rev']
    r['auc_sep_rule'] = r['auc_fwd'] if ch_sep == 'forward' else r['auc_rev']
    r['auc_oracle'] = max(r['auc_fwd'], r['auc_rev'])
    r['ci95'] = M.auc_ci(p, label, sup, n=300)
    # elongation on predictions zeroed outside the support, so every arm (direct or mapped) sees the same context
    r['elong'] = M.elongation(np.where(sup, p, 0.0).astype(np.float32), label, sup, PIX)
    if c2:
        fl = M.shuffle_floor(p, label, sup, seeds=8)
        r['c2_median'] = float(np.median(fl))
    return r, p


def stage_score(tags):
    res = dict(settings=dict(band=[ZLO, ZHI], n_slices=NS, ckpt=CKPT, rcov=RCOV, rmax=RMAX, pixel_um=PIX), windings={})
    pooled = {}
    for w in WINDS:
        g = np.load(W / 'org' / w / 'geo.npz')
        band, label = g['band'], g['label'].astype(bool) & g['band']
        r0, r1, c0, c1 = g['sub']
        H, Wd = band.shape
        Q = g['pts'][band]
        R = {}
        # A: organiser render (E1 predictions) on the same pixels
        a_f = _pred(E1 / f'1447_{w}' / 'pred' / 's42_A.tif')[r0:r1, c0:c1]
        a_r = _pred(E1 / f'1447_{w}' / 'pred' / 's42_A_reverse.tif')[r0:r1, c0:c1]
        R['A'], pA = score_canvas(a_f, a_r, label, band, order='reverse')
        canv = {'A': pA}
        v8 = E1 / f'1447_{w}' / 'v8in_released.png'
        if v8.exists():   # organisers' released v8-in prediction on the organiser render (already depth-oriented)
            import cv2
            pv = cv2.imread(str(v8), cv2.IMREAD_GRAYSCALE).astype(np.float32)[r0:r1, c0:c1] / 255.0
            R['V8'], canv['V8'] = score_canvas(pv, pv, label, band)
        for direct in ('Asub', 'R'):
            if (ARMS / f'{w}_{direct}_pred_reverse.tif').exists():
                rf, rr = load_pair(f'{w}_{direct}')
                R[direct], canv[direct] = score_canvas(rf, rr, label, band, order='reverse')
        for f in sorted(ARMS.glob(f'{w}_OFS*_pred_reverse.tif')):   # organiser layout + auto offset field (direct)
            k = f.name[len(w) + 1:-len('_pred_reverse.tif')]
            rf, rr = load_pair(f'{w}_{k}')
            R[k], canv[k] = score_canvas(rf, rr, label, band, c2=False, order='reverse')
        for dr, dc in PHASES:
            k = f'Rph{dr}_{dc}'
            if (ARMS / f'{w}_{k}_pred_reverse.tif').exists():
                rf, rr = load_pair(f'{w}_{k}')
                R[k], canv[k] = score_canvas(rf[dr:, dc:], rr[dr:, dc:], label, band, c2=False, order='reverse')
        cand = ['R', 'P10', 'M10', 'X1'] + [k for k, _ in DOSE]
        arms = [a for a in cand if (ARMS / f'{(w if a != "X1" else [x for x in WINDS if x != w][0])}_{"R" if a == "X1" else a}_pred_reverse.tif').exists()]
        arms += [f'auto_{t}' for t in tags]
        for arm in arms:
            t = time.time()
            pos, pf, pr, idx, nrm_arm = surface_arm(arm, w)
            lo, hi = Q.min(0) - RMAX, Q.max(0) + RMAX
            sel = np.all((pos >= lo) & (pos <= hi), 1)
            T = cKDTree(pos[sel], balanced_tree=False, compact_nodes=False)
            d, j = T.query(Q, distance_upper_bound=RMAX, workers=4)
            hit = np.isfinite(d)
            jf = np.where(hit, j, 0)
            mf = np.where(hit, pf[sel][jf], 0.0); mr = np.where(hit, pr[sel][jf], 0.0)
            key = arm + ('m' if arm == 'R' else '')
            fixed = 'forward' if arm.startswith('auto_') else 'reverse'
            var = {}
            for vname, cap in [('e2e_c14', RCOV), ('e2e_c6', RCOV_STRICT), ('near60', RMAX)]:
                cov = hit & (d <= cap)
                CF = np.zeros((H, Wd), np.float32); CR = np.zeros((H, Wd), np.float32)
                CF[band] = np.where(cov, mf, 0.0); CR[band] = np.where(cov, mr, 0.0)
                var[vname] = score_canvas(CF, CR, label, band, c2=(vname == 'e2e_c14'), order=fixed)
                if vname == 'e2e_c14':
                    covm = np.zeros((H, Wd), bool); covm[band] = cov
                    r_cov = score_canvas(CF, CR, label, covm, c2=False, order=fixed)[0] if cov.sum() > 1000 else {}
            r_e2e, pE = var['e2e_c14']
            dd = d[hit]
            adh = dict(frac_le3=float((hit & (d <= 3)).mean()), frac_le6=float((hit & (d <= 6)).mean()),
                       frac_le10=float((hit & (d <= 10)).mean()), frac_le20=float((hit & (d <= 20)).mean()),
                       frac_found_le60=float(hit.mean()), d_median=float(np.median(dd)) if dd.size else None,
                       d_p95=float(np.percentile(dd, 95)) if dd.size else None)
            m14 = hit & (d <= RCOV)
            if m14.sum() > 100:
                cosang = np.abs((g['nrm'][band][m14] * nrm_arm[sel][jf[m14]]).sum(1)).clip(0, 1)
                ang = np.degrees(np.arccos(cosang))
                adh['normal_angle_deg_median'] = float(np.nanmedian(ang)); adh['normal_angle_deg_p90'] = float(np.nanpercentile(ang, 90))
            if idx is not None and arm != 'X1':
                # tangential slip: the true correspondence is the same canvas pixel
                q_rc = np.argwhere(band); src = idx[sel][jf]
                slip = np.linalg.norm((src - q_rc).astype(np.float32), axis=1)[hit]
                adh['tangential_slip_px_median'] = float(np.median(slip)); adh['tangential_slip_px_p95'] = float(np.percentile(slip, 95))
            R[key] = dict(end_to_end=r_e2e, covered=r_cov, e2e_c6=var['e2e_c6'][0], near60=var['near60'][0], adherence=adh,
                          secs=round(time.time() - t, 1))
            canv[key] = pE
            print(w, key, 'AUC e2e@14 %.4f e2e@6 %.4f near60 %.4f covered %s  le6 %.3f  d_med %s  ang_med %s' % (
                r_e2e['auc'], var['e2e_c6'][0]['auc'], var['near60'][0]['auc'], ('%.4f' % r_cov['auc']) if r_cov else '-',
                adh['frac_le6'], adh['d_median'], adh.get('normal_angle_deg_median')), flush=True)
        # paired deltas vs A on the same support (band)
        for k, p in canv.items():
            if k == 'A':
                continue
            ha, hb = block_pair(canv['A'], p, label, band)
            R[k]['dAUC_vs_A'] = (M.auc_from(hb[0].sum(0), hb[1].sum(0)) - M.auc_from(ha[0].sum(0), ha[1].sum(0)))
            R[k]['dAUC_vs_A_ci95'] = paired_ci(ha, hb)
            pooled.setdefault(k, []).append(hb); pooled.setdefault('A', [])
        pooled['A'].append(M.block_hists(M.to_u8(canv['A']), label, band))
        res['windings'][w] = R
    # pooled (w058 + w060) AUC per arm with paired CI vs A
    res['pooled'] = {}
    hA = (np.concatenate([h[0] for h in pooled['A']]), np.concatenate([h[1] for h in pooled['A']]))
    for k, hs in pooled.items():
        if len(hs) != len(WINDS):
            continue
        h = (np.concatenate([x[0] for x in hs]), np.concatenate([x[1] for x in hs]))
        rec = dict(auc=M.auc_from(h[0].sum(0), h[1].sum(0)))
        if k != 'A':
            rec['dAUC_vs_A'] = rec['auc'] - M.auc_from(hA[0].sum(0), hA[1].sum(0)); rec['ci95'] = paired_ci(hA, h)
        res['pooled'][k] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'g2_scores.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res['pooled'], indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage'); ap.add_argument('rest', nargs='*'); ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--noflip', action='store_true')
    a = ap.parse_args()
    if a.stage == 'org':
        stage_org(a.procs)
    elif a.stage == 'auto':
        stage_auto(a.rest[0], a.rest[1], a.procs, flip=not a.noflip)
    elif a.stage == 'phase':
        stage_phase()
    elif a.stage == 'asub':
        stage_asub()
    elif a.stage == 'ofs':
        stage_ofs(a.rest[0], a.procs)
    elif a.stage == 'infercpu':
        stage_infer_cpu(a.rest)
    elif a.stage == 'infer':
        stage_infer(a.rest)
    elif a.stage == 'score':
        stage_score(a.rest)


if __name__ == '__main__':
    main()
