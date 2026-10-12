#!/usr/bin/env python
"""band_screen: JOB 3 driver for one fitted + flattened band. Only the validated tools/ensemble code does
the science (render66 through r66.py, depth_sweep through sweep.py, ensemble_maps.py, display.py's compose style);
this file adds strip bookkeeping, a simple contrast score, the selection and resumability.

  plan    canvas size, column strips with overlap, ONE normal sign for the whole surface (umbilicus outward test,
          refuses an ambiguous mesh), a CT chunk estimate.
  screen  per strip: render the layers dense_native needs for d in {-2, 0, +2} (k 23..43), sweep dense_native on the
          reading side (reverse) at those depths, keep the maps + valid mask, delete the render (intermediate).
          Optional look-ahead: the next strip renders (CPU) while this one sweeps (GPU). Cost guard before each strip.
  select  canvas tiles of TILE x TILE px with >= MIN_VALID valid pixels; contrast = p95 - p50 of the depth-mean
          dense_native unit map (villa to_unit) over the tile's valid pixels; the top TOP_FRAC of valid tiles by contrast
          are selected (tiles.csv keeps p50/p95/p99/mean of every tile, so a different rule can be applied later).
  ens     per strip with selected tiles: render k 19..47 only on the selected tiles + CTX px of context (pixel mask),
          the 3 villa members at d = -6..+6 step 2 on the selected pixels (need mask), ensemble_maps z-mean with the
          mean over depth ('dmean'), then his dimmed-CT yellow style (display.compose, CT at d = +1) per cluster of
          adjacent selected tiles: one crop PNG + line-scale strips (450 px, 25 % overlap, 1 cm bar).
Kept: plan/tiles/selection/index JSON+CSV, all model maps (uint8 tif), valid masks and dmean (compressed npz), PNGs.
Pruned (regenerable): strip renders, per-depth ensemble .npy (rebuilt from the kept maps by ensemble_maps.py).
Env: BS_RENDER_CMD / BS_SWEEP_CMD replace the render / sweep command prefix (dry-run mocks), POD_STATE for the guard
(default $VC_ROOT/cloud_state), ENS_LANE for the ensemble tools (default tools/ensemble), POD_PY for the python used by
the sub-commands.
"""
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]                                   # tools/: vcpaths.py, g12lib.py, ensemble/
sys.path.insert(0, str(TOOLS))
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

ROOT = Path(_P.ROOT)
ENS = Path(os.environ.get('ENS_LANE', str(TOOLS / 'ensemble')))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENS))
PY = os.environ.get('POD_PY', sys.executable)
SCREEN_K = '23:44'          # villa window for d in -2..+2 = layers 25+d-2 .. 41+d  -> 23..43
ENS_K = '19:48'             # d in -6..+6 -> 19..47
VILLA_FALLBACK_S_PER_MPX = 15.3      # pessimistic RTX 4090 guess, used until timing.json exists


def log(msg):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} [band_screen] {msg}', flush=True)


def rng(s):
    a, b = s.split(':')
    return int(a), int(b)


def guard(label, est_h):
    """cost_guard check; returns its exit code (0 = go)."""
    cg = HERE / 'cost_guard.py'
    cmd = [PY, str(cg), 'check', label] + ([] if est_h is None else ['--est-h', f'{est_h:.3f}'])
    return subprocess.run(cmd).returncode


def s_per_mpx(model):
    """Measured 4090 seconds per valid Mpx per depth (job 1 timing.json), else the cost-model guess."""
    tj = Path(os.environ.get('POD_STATE', ROOT / 'cloud_state')) / 'timing.json'
    if tj.exists():
        t = json.loads(tj.read_text()).get('villa_s_per_valid_mpx', {})
        if model in t:
            return float(t[model])
        if t:
            return float(max(t.values()))
    return VILLA_FALLBACK_S_PER_MPX


def run(cmd, logf, env=None):
    with open(logf, 'a') as f:
        f.write(f'\n$ {" ".join(map(str, cmd))}\n'); f.flush()
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def render_cmd():
    return shlex.split(os.environ.get('BS_RENDER_CMD', f'{PY} {HERE / "r66.py"}'))


def sweep_cmd():
    return shlex.split(os.environ.get('BS_SWEEP_CMD', f'{PY} {HERE / "sweep.py"}'))


# ------------------------------------------------------------------------------------------------------------ plan
def cmd_plan(a):
    import volumes
    C = volumes.register()
    sys.path.insert(0, C.G12)
    import g12lib as L
    P, valid, meta = L.load_tifxyz(a.tifxyz)
    gs = float(meta['scale'][0])
    H = int(np.floor(P.shape[0] / gs)); W = int(np.floor(P.shape[1] / gs))
    r0, r1 = rng(a.rows) if a.rows else (0, H)
    c0, c1 = rng(a.cols) if a.cols else (0, W)
    if a.strip_cols % a.tile:
        sys.exit(f'--strip-cols {a.strip_cols} must be a multiple of --tile {a.tile}')
    pxv = (1.0 / gs) ** 2
    gr0, gr1 = int(np.floor(r0 * gs)), int(np.ceil(r1 * gs))
    strips = []
    for s in range(c0, c1, a.strip_cols):
        e = min(c1, s + a.strip_cols)
        nv = int(valid[gr0:gr1, int(np.floor(s * gs)):int(np.ceil(e * gs))].sum())
        if nv == 0:
            continue
        strips.append(dict(i=len(strips), keep=[s, e], render=[max(c0, s - a.overlap), min(c1, e + a.overlap)],
                           valid_px_est=int(nv * pxv)))
    N = L.vertex_normals(P)
    umb = C.load_umbilicus(a.umbilicus or C.VOLUMES[a.scroll]['umbilicus'])
    fout, npx = C.outward_fraction(P, N, umb)
    if 0.2 < fout < 0.8 and not a.force:
        print(json.dumps(dict(error='normal sign ambiguous', outward_fraction=fout)))
        return 3
    sign = 'flip' if fout < 0.5 else 'keep'
    # CT estimate: chunks touched by the grid vertices at the deepest offsets of the ensemble window (k 19..47 ->
    # -13.5..+14.5 vox) and the surface; vertex spacing 1/gs px (20 vox) < 128, so the set is dense. x1.15 for the
    # per-tile read-box margins of the renderer.
    v = valid & np.isfinite(N).all(-1)
    p = P[v].astype(np.float64); nn = N[v].astype(np.float64)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
    shp = C.VOLUMES[a.scroll]['shape']
    keys = []
    for o in (-13.5, 0.0, 14.5):
        q = np.floor((p + o * nn) / 128).astype(np.int64)                # XYZ chunk index
        ok = (q >= 0).all(1) & (q[:, 2] * 128 < shp[0]) & (q[:, 1] * 128 < shp[1]) & (q[:, 0] * 128 < shp[2])
        q = q[ok]
        keys.append(q[:, 2] * 10 ** 8 + q[:, 1] * 10 ** 4 + q[:, 0])
    nk = int(np.unique(np.concatenate(keys)).size) if keys else 0
    plan = dict(scroll=a.scroll, tifxyz=str(Path(a.tifxyz).resolve()), grid_scale=gs, canvas_hw=[H, W],
                rows=[r0, r1], cols=[c0, c1], strip_cols=a.strip_cols, overlap=a.overlap, tile=a.tile,
                strips=strips, normal_sign=sign, outward_fraction=fout, outward_px=npx,
                valid_mpx_est=round(float(valid[gr0:gr1, int(np.floor(c0 * gs)):int(np.ceil(c1 * gs))].sum()) * pxv / 1e6, 2),
                area_cm2_meta=meta.get('area_cm2'), ct_chunks_est=nk, voxel_um=C.VOLUMES[a.scroll].get('um', 8.64),
                ct_gb_est=round(nk * 128 ** 3 / 1e9 * 1.15, 1), out=str(Path(a.out).resolve()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: v for k, v in plan.items() if k != 'strips'} | dict(n_strips=len(strips))))
    return 0


# ---------------------------------------------------------------------------------------------------------- screen
def strip_dir(plan, i):
    return Path(plan['out']) / 'strips' / f'{i:04d}'


def start_render(plan, st, kr, zarr, logf, pixel_mask=None, procs=3, fetch=True):
    if Path(zarr).exists():
        return None
    r0, r1 = plan['rows']; c0, c1 = st['render']
    cmd = render_cmd() + ['--scroll', plan['scroll'], '--tifxyz', plan['tifxyz'], '--rows', f'{r0}:{r1}',
                          '--cols', f'{c0}:{c1}', '--normal-sign', plan['normal_sign'], '--k-range', kr,
                          '--out', str(zarr), '--procs', str(procs), '--tile', '64']
    if pixel_mask is not None:
        cmd += ['--pixel-mask', str(pixel_mask)]
    env = dict(os.environ, OMP_NUM_THREADS='1')
    if fetch:
        env['ENS_ALLOW_FETCH'] = '1'
    f = open(logf, 'a'); f.write(f'\n$ {" ".join(cmd)}\n'); f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def save_valid(zarr, dst):
    from ens import common as C
    arr, at = C.open_render(zarr)
    v = C.valid_mask_all_layers(arr, k_range=at.get('ensemble_render', {}).get('rendered_k'))
    np.savez_compressed(dst, valid=v)
    return int(v.sum())


def cmd_screen(a):
    plan = json.loads(Path(a.plan).read_text())
    strips = plan['strips']
    if a.strip_range:
        s0, s1 = rng(a.strip_range); strips = [s for s in strips if s0 <= s['i'] < s1]
    todo = [s for s in strips if not (strip_dir(plan, s['i']) / 'screen.done').exists()]
    log(f"{plan['scroll']}: {len(strips) - len(todo)} of {len(strips)} strips already screened; {len(todo)} to go")
    spm = s_per_mpx(a.model); nd = n_depths(a.depths)
    pending = {}
    for idx, st in enumerate(todo):
        if a.max_strips is not None and idx >= a.max_strips:
            log('max-strips reached'); break
        sd = strip_dir(plan, st['i']); sd.mkdir(parents=True, exist_ok=True)
        lg = sd / 'screen.log'; z = sd / 'r_screen.zarr'
        est_h = st['valid_px_est'] / 1e6 * nd * spm / 3600
        rc = guard(f"screen {plan['scroll']} strip {st['i'] + 1}/{len(plan['strips'])}", est_h)
        if rc:
            for p in pending.values():
                if p:
                    p.terminate()
            return rc
        t0 = time.time()
        p = pending.pop(st['i'], None)
        if p is None and not z.exists():
            p = start_render(plan, st, SCREEN_K, z, lg, procs=a.procs, fetch=not a.no_fetch)
        if p is not None and p.wait() != 0:
            log(f"render failed for strip {st['i']} (see {lg})"); return 2
        t_render = time.time() - t0
        if a.pipeline and idx + 1 < len(todo):
            nx = todo[idx + 1]; nsd = strip_dir(plan, nx['i']); nsd.mkdir(parents=True, exist_ok=True)
            pending[nx['i']] = start_render(plan, nx, SCREEN_K, nsd / 'r_screen.zarr', nsd / 'screen.log',
                                            procs=a.procs, fetch=not a.no_fetch)
        t1 = time.time()
        cmd = sweep_cmd() + ['--render', z, '--models', a.model, f'--depths={a.depths}', '--order', 'reverse',
                             '--phase', '0,0', '--out', sd / 'sw', '--device', a.device, '--tile', '2048']
        if a.batch:
            cmd += ['--batch', str(a.batch)]
        if a.threads:
            cmd += ['--threads', str(a.threads)]
        if run(cmd, lg) != 0:
            log(f"sweep failed for strip {st['i']} (see {lg})"); return 2
        t_sweep = time.time() - t1
        nvalid = save_valid(z, sd / 'valid.npz')
        if not a.keep_renders:
            shutil.rmtree(z, ignore_errors=True)
        rec = dict(strip=st['i'], render_s=round(t_render, 1), sweep_s=round(t_sweep, 1), valid_px=nvalid,
                   s_per_valid_mpx_depth=round(t_sweep / max(nvalid / 1e6, 1e-9) / nd, 2), model=a.model,
                   depths=a.depths)
        (sd / 'screen.done').write_text(json.dumps(rec))
        log(f"strip {st['i']}: render {t_render:.0f}s (overlapped: {a.pipeline}), sweep {t_sweep:.0f}s, "
            f"{nvalid / 1e6:.2f} valid Mpx, {rec['s_per_valid_mpx_depth']} s/Mpx/depth")
    return 0


def n_depths(spec):
    from ens import common as C
    return len(C.parse_depths(spec))


# ---------------------------------------------------------------------------------------------------------- select
def unit_mean(sd, model, depths):
    import tifffile
    from ens import adapters as A, common as C
    acc = None; n = 0
    for d in C.parse_depths(depths):
        m = tifffile.imread(sd / 'sw' / 'maps' / model / f'{model}__reverse__ph0_0__d{d:+03d}.tif')
        u = A.VillaAdapter.to_unit(m)
        acc = u if acc is None else acc + u; n += 1
    return acc / n


def cmd_select(a):
    plan = json.loads(Path(a.plan).read_text())
    T = plan['tile']; r0 = plan['rows'][0]; c0 = plan['cols'][0]
    rows = []
    missing = [s['i'] for s in plan['strips'] if not (strip_dir(plan, s['i']) / 'screen.done').exists()]
    if missing and not a.partial:
        log(f'{len(missing)} strips not screened yet (e.g. {missing[:5]}); use --partial to select anyway'); return 2
    for st in plan['strips']:
        sd = strip_dir(plan, st['i'])
        if not (sd / 'screen.done').exists():
            continue
        u = unit_mean(sd, a.model, a.depths)
        v = np.load(sd / 'valid.npz')['valid']
        if a.edge_erode > 0:   # interior = valid pixels at least edge_erode px (Chebyshev) from the mask edge
            from scipy import ndimage
            ve = ndimage.minimum_filter(v.astype(np.uint8), size=2 * a.edge_erode + 1, mode='constant', cval=0) > 0
        else:
            ve = v
        rc0 = st['render'][0]
        k0, k1 = st['keep'][0] - rc0, st['keep'][1] - rc0
        H = u.shape[0]
        for ty in range(0, H, T):
            for gx in range(st['keep'][0], st['keep'][1], T):
                lx = gx - rc0
                vv = v[ty:ty + T, lx:min(lx + T, k1)]
                if vv.size == 0:
                    continue
                vf = float(vv.mean())
                if vf < a.min_valid:
                    continue
                ee = ve[ty:ty + T, lx:min(lx + T, k1)]
                if ee.mean() < 0.25:
                    continue
                x = u[ty:ty + T, lx:min(lx + T, k1)][ee]
                p50, p95, p99 = (float(q) for q in np.percentile(x, [50, 95, 99]))
                rows.append(dict(strip=st['i'], tr=(ty) // T, tc=(gx - c0) // T, row0=r0 + ty, col0=gx,
                                 valid_frac=round(vf, 3), score=round(p95 - p50, 5), p50=round(p50, 4),
                                 p95=round(p95, 4), p99=round(p99, 4), mean=round(float(x.mean()), 4)))
    if not rows:
        log('no valid tiles'); return 2
    order = sorted(range(len(rows)), key=lambda i: -rows[i]['score'])
    nsel = max(1, math.ceil(a.top_frac * len(rows)))
    for r in rows:
        r['selected'] = False
    for i in order[:nsel]:
        rows[i]['selected'] = True
    out = Path(plan['out'])
    import csv
    tg = a.tag or ''
    with open(out / f'tiles{tg}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sel = [r for r in rows if r['selected']]
    res = dict(rule=f'contrast = p95 - p50 of the mean over d={a.depths} of {a.model} unit maps (villa to_unit) on '
                    f'valid px >= {a.edge_erode} px inside the mask edge (>= 25 % of the tile); tiles {T} px with '
                    f'>= {a.min_valid:.0%} valid; top {a.top_frac:.0%} of valid tiles',
               n_valid_tiles=len(rows), n_selected=len(sel), partial=bool(missing), score_threshold=min(r['score'] for r in sel) if sel else None,
               selected=sel)
    (out / f'selection{tg}.json').write_text(json.dumps(res, indent=1))
    overview(rows, plan, out / f'overview_scores{tg}.png')
    log(f"{plan['scroll']}: {len(rows)} valid tiles, {len(sel)} selected (score >= {res['score_threshold']})")
    return 0


def overview(rows, plan, path, wrap=512):
    """Tile-resolution contrast map (grey), selected tiles in yellow, wrapped every `wrap` tile columns."""
    import cv2
    T = plan['tile']
    nr = -(-(plan['rows'][1] - plan['rows'][0]) // T); nc = -(-(plan['cols'][1] - plan['cols'][0]) // T)
    s = np.array([r['score'] for r in rows]); lo, hi = np.percentile(s, [1, 99.5]) if len(s) > 1 else (0, 1)
    img = np.zeros((nr, nc, 3), np.uint8)
    for r in rows:
        g = int(255 * np.clip((r['score'] - lo) / max(hi - lo, 1e-9), 0, 1))
        img[r['tr'], r['tc']] = (120, 235, 255) if r['selected'] else (g, g, g)
    bands = [img[:, i:i + wrap] for i in range(0, nc, wrap)]
    bands = [np.pad(b, ((0, 1), (0, wrap - b.shape[1]), (0, 0))) for b in bands]
    big = np.concatenate(bands, 0)
    big = cv2.resize(big, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), big)


# ------------------------------------------------------------------------------------------------------------- ens
def clusters(tiles):
    """8-connected components of (tr, tc) tile cells -> list of lists."""
    left = set((t['tr'], t['tc']) for t in tiles); by = {(t['tr'], t['tc']): t for t in tiles}; out = []
    while left:
        stack = [left.pop()]; comp = []
        while stack:
            c = stack.pop(); comp.append(by[c])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (c[0] + dy, c[1] + dx)
                    if n in left:
                        left.remove(n); stack.append(n)
        out.append(comp)
    return out


def draw(zarr, ens_dir, name, comps, plan, st, out_png, ct_d=1, strip_h=450, voxel_um=8.64):
    """display.py's style in-process (compose + scale bar + line-scale strips), one crop per cluster."""
    import cv2
    import display as D
    from ens import common as C
    voxel_um = plan.get('voxel_um', voxel_um)      # scale bars use the scroll's voxel size
    arr, at = C.open_render(zarr)
    n = arr.shape[0]; mid = n // 2
    stt = json.loads((Path(ens_dir) / f'{name}__stats.json').read_text())
    valid = np.load(Path(ens_dir) / f'{name}__valid.npy')
    X = np.load(Path(ens_dir) / f'{name}__ensemble__dmean.npy')
    lo, hi = stt['ensemble_dmean']['p1'], stt['ensemble_dmean']['p99_5']
    ink = (np.clip((X - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8); ink[~valid] = 0
    ctvalid = C.valid_mask_all_layers(arr, k_range=at['ensemble_render'].get('rendered_k'))
    ctl = np.asarray(arr[mid + ct_d]).astype(np.float32)
    ct = (np.clip((ctl - stt['ct']['p1']) / max(stt['ct']['p99_5'] - stt['ct']['p1'], 1e-9), 0, 1) * 255).astype(np.uint8)
    img = D.compose(ct, ink, ctvalid)
    T = plan['tile']; r0g = plan['rows'][0]; rc0 = st['render'][0]
    out_png.mkdir(parents=True, exist_ok=True)
    recs = []
    for k, comp in enumerate(sorted(comps, key=lambda c: -max(t['score'] for t in c))):
        ys = [t['row0'] - r0g for t in comp]; xs = [t['col0'] - rc0 for t in comp]
        y0, y1 = max(0, min(ys)), min(img.shape[0], max(ys) + T)
        x0, x1 = max(0, min(xs)), min(img.shape[1], max(xs) + T)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        tag = f"{plan['scroll']}_s{st['i']:04d}_c{k:02d}"
        cv2.imwrite(str(out_png / f'{tag}.png'), D.scale_bar(crop.copy(), voxel_um))
        sdir = out_png / f'{tag}_strips'; sdir.mkdir(exist_ok=True)
        step = max(1, int(strip_h * 0.75)); j = 0
        for s0 in range(0, max(1, crop.shape[0] - strip_h // 4), step):
            cv2.imwrite(str(sdir / f'strip_{j:02d}_r{s0}.png'), D.scale_bar(np.ascontiguousarray(crop[s0:s0 + strip_h]), voxel_um))
            j += 1
        recs.append(dict(cluster=tag, strip=st['i'], n_tiles=len(comp), max_score=max(t['score'] for t in comp),
                         mean_score=round(float(np.mean([t['score'] for t in comp])), 5),
                         canvas_rows=[r0g + y0, r0g + y1], canvas_cols=[rc0 + x0, rc0 + x1],
                         png=f'strips/{tag}.png', strips=f'strips/{tag}_strips', n_strips=j))   # relative to results/<SCROLL>
    return recs


def cmd_ens(a):
    plan = json.loads(Path(a.plan).read_text())
    sel = json.loads((Path(plan['out']) / 'selection.json').read_text())['selected']
    T = plan['tile']; r0g, r1g = plan['rows']
    by = {}
    for t in sel:
        by.setdefault(t['strip'], []).append(t)
    models = a.models.split(',')
    nd = n_depths(a.depths)
    sumspm = sum(s_per_mpx(m) for m in models)
    res_dir = Path(a.results); res_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for st in plan['strips']:
        if st['i'] not in by:
            continue
        sd = strip_dir(plan, st['i']); sd.mkdir(parents=True, exist_ok=True)
        done = sd / 'ens.done'
        if done.exists():
            index += json.loads(done.read_text())['clusters']; continue
        tiles = by[st['i']]
        lg = sd / 'ens.log'
        H = r1g - r0g; rc0, rc1 = st['render']; Wr = rc1 - rc0
        need = np.zeros((H, Wr), bool)
        for t in tiles:
            y, x = t['row0'] - r0g, t['col0'] - rc0
            need[y:y + T, x:x + T] = True
        pm = np.zeros_like(need)
        for t in tiles:
            y, x = t['row0'] - r0g, t['col0'] - rc0
            pm[max(0, y - a.ctx):y + T + a.ctx, max(0, x - a.ctx):x + T + a.ctx] = True
        est_h = need.sum() * 1.4 / 1e6 * nd * sumspm / 3600
        rc = guard(f"ens {plan['scroll']} strip {st['i']} ({len(tiles)} tiles)", est_h)
        if rc:
            return rc
        np.save(sd / 'ens_pixmask.npy', pm); np.save(sd / 'ens_need.npy', need)
        z = sd / 'r_ens.zarr'
        p = start_render(plan, st, ENS_K, z, lg, pixel_mask=sd / 'ens_pixmask.npy', procs=a.procs, fetch=not a.no_fetch)
        if p is not None and p.wait() != 0:
            log(f"ens render failed strip {st['i']} ({lg})"); return 2
        cmd = sweep_cmd() + ['--render', z, '--models', a.models, f'--depths={a.depths}', '--order', 'reverse',
                             '--phase', '0,0', '--need-mask', sd / 'ens_need.npy', '--out', sd / 'ens_sw',
                             '--device', a.device, '--tile', '2048']
        if a.batch:
            cmd += ['--batch', str(a.batch)]
        if a.threads:
            cmd += ['--threads', str(a.threads)]
        t1 = time.time()
        if run(cmd, lg) != 0:
            log(f"ens sweep failed strip {st['i']} ({lg})"); return 2
        t_sw = time.time() - t1
        name = f"{plan['scroll']}_s{st['i']:04d}"
        if run([PY, ENS / 'ensemble_maps.py', '--render', z, '--maps', sd / 'ens_sw', '--models', a.models,
                '--tag', 'ph0_0', f'--stat-depths={a.depths}', f'--write-depths={a.depths}', '--combine', 'mean',
                '--out', sd / 'ens', '--name', name, '--no-png', '--stat-mask', sd / 'ens_need.npy'], lg) != 0:
            log(f"ensemble_maps failed strip {st['i']} ({lg})"); return 2
        recs = draw(z, sd / 'ens', name, clusters(tiles), plan, st, res_dir / 'strips', ct_d=a.ct_d, strip_h=a.strip_h)
        # keep dmean + valid compressed; per-depth ensemble npy are regenerable from the kept maps
        e = sd / 'ens'
        np.savez_compressed(e / f'{name}__dmean_f16.npz', dmean=np.load(e / f'{name}__ensemble__dmean.npy').astype(np.float16),
                            valid=np.load(e / f'{name}__valid.npy'))
        for f in e.glob(f'{name}__ensemble__*.npy'):
            f.unlink()
        (e / f'{name}__valid.npy').unlink(missing_ok=True)
        if not a.keep_renders:
            shutil.rmtree(z, ignore_errors=True)
        done.write_text(json.dumps(dict(strip=st['i'], sweep_s=round(t_sw, 1), need_px=int(need.sum()), clusters=recs)))
        index += recs
        log(f"ens strip {st['i']}: {len(tiles)} tiles, {len(recs)} clusters, sweep {t_sw:.0f}s")
    index.sort(key=lambda r: -r['max_score'])
    (res_dir / 'index.json').write_text(json.dumps(index, indent=1))
    import csv
    if index:
        with open(res_dir / 'index.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[k for k in index[0] if k not in ('canvas_rows', 'canvas_cols')] +
                               ['r0', 'r1', 'c0', 'c1'])
            w.writeheader()
            for r in index:
                rr = {k: v for k, v in r.items() if k not in ('canvas_rows', 'canvas_cols')}
                rr.update(r0=r['canvas_rows'][0], r1=r['canvas_rows'][1], c0=r['canvas_cols'][0], c1=r['canvas_cols'][1])
                w.writerow(rr)
    log(f"{plan['scroll']}: {len(index)} clusters drawn -> {res_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--scroll', required=True); p.add_argument('--tifxyz', required=True); p.add_argument('--out', required=True)
    p.add_argument('--strip-cols', type=int, default=16384); p.add_argument('--overlap', type=int, default=128)
    p.add_argument('--tile', type=int, default=512); p.add_argument('--rows'); p.add_argument('--cols')
    p.add_argument('--umbilicus'); p.add_argument('--force', action='store_true')
    for name in ('screen', 'select', 'ens'):
        q = sub.add_parser(name); q.add_argument('--plan', required=True)
        q.add_argument('--device', default='cuda'); q.add_argument('--batch', type=int, default=0)
        q.add_argument('--threads', type=int, default=0); q.add_argument('--procs', type=int, default=3)
        q.add_argument('--keep-renders', action='store_true'); q.add_argument('--no-fetch', action='store_true')
        q.add_argument('--model', default='dnative'); q.add_argument('--depths', default='-2..2:2' if name != 'ens' else '-6..6:2')
    sp = sub.choices['screen']
    sp.add_argument('--pipeline', type=int, default=1); sp.add_argument('--max-strips', type=int)
    sp.add_argument('--strip-range')
    se = sub.choices['select']
    se.add_argument('--top-frac', type=float, default=0.10); se.add_argument('--min-valid', type=float, default=0.5)
    se.add_argument('--partial', action='store_true')
    se.add_argument('--edge-erode', type=int, default=int(os.environ.get('BS_EDGE_ERODE', '128')))
    se.add_argument('--tag', default='')
    en = sub.choices['ens']
    en.add_argument('--models', default='ink9_s42,ink9_s43,dnative'); en.add_argument('--ctx', type=int, default=128)
    en.add_argument('--results', required=True); en.add_argument('--ct-d', type=int, default=1)
    en.add_argument('--strip-h', type=int, default=450)
    a = ap.parse_args()
    return dict(plan=cmd_plan, screen=cmd_screen, select=cmd_select, ens=cmd_ens)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
