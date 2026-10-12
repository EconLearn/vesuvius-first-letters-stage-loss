#!/usr/bin/env python
"""depth_sweep: run ink models at a list of depths d on a 66-layer render (render66.py), resumable per depth and per
tile, with tile-phase runs and merge. Reproduces Nieuwlaar's run_depth_sweep.sh + run_ink.sh + pad_render.py +
merge_tile_phase.py inside our pipeline (see ens/adapters.py for each model's exact window and tiling).

  depth_sweep.py --render R.zarr --models ink9_s42,ink9_s43,dnative,hecate --depths=-6..6:2 --order reverse \
                 --phase 0,0 --phase 40,200 --merge mean --out DIR [--device cuda]

* depths are visited centre-out (0, -1, +1, ...), so a partial run already covers the central depths;
* --order reverse|forward|both (reading side in his convention = reverse);
* --phase PR,PC (repeatable): virtual zero padding of PR rows on top and PC columns on the left (his pad_render.py,
  default 40,200), predicted on the GLOBAL patch grid of the padded canvas, cropped back. 0,0 = the plain run;
* --merge none|mean|mask:FILE.npy  (FILE = bool H x W render-grid mask, True -> take the second phase there; his
  merge takes the w047 half from the padded run). Output tag 'merged';
* outputs: DIR/maps/<model>/<model>__<order>__<tag>__d+NN.tif (+ .json); timings in DIR/timings.tsv;
  per-tile partials in DIR/.tiles (deleted when the map is assembled);
* GPU: --device cuda refuses to start unless launched through sweep_locked.sh (GPU_LOCK held, no fit running).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens import adapters as A  # noqa: E402
from ens import common as C  # noqa: E402


def dtag(d):
    return f'd{int(d):+03d}'


def map_path(out, model, order, tag, d):
    return Path(out) / 'maps' / model / f'{model}__{order}__{tag}__{dtag(d)}.tif'


def gpu_guard(device):
    if not str(device).startswith('cuda'):
        return
    if os.environ.get('ENS_GPU_LOCK_HELD') != '1':
        sys.exit('refusing GPU run: launch through sweep_locked.sh (holds the GPU lock and '
                 'waits until no fit_spiral.py runs)')
    if subprocess.run(['pgrep', '-f', 'fit_spiral.py'], capture_output=True).stdout.strip():
        sys.exit('refusing GPU run: a fit_spiral.py process is running')


def need_in_grid(ad, need, pr, pc, Hm, Wm):
    """Render-grid bool need mask -> padded model grid."""
    npd = np.zeros((need.shape[0] + pr, need.shape[1] + pc), bool); npd[pr:, pc:] = need
    if npd.shape != (Hm, Wm):
        import cv2
        npd = cv2.resize(npd.astype(np.uint8), (Wm, Hm), interpolation=cv2.INTER_NEAREST).astype(bool)
    return npd


def run_one(ad, src, d, order, phase, out, tile, batch, log, max_tiles=None, need=None):
    """One (model, depth, order, phase) map, tile-resumable. Returns the map path.
    need (bool, render grid): only patches touching need pixels are run; the map is exact on need pixels (every
    patch covering them runs) and set to 0 elsewhere."""
    pr, pc = phase
    tag = f'ph{pr}_{pc}'
    mp = map_path(out, ad.name, order, tag, d)
    if mp.exists():
        return mp, 'exists'
    Hp, Wp = src.H + pr, src.W + pc
    Hm, Wm = ad.grid_hw(Hp, Wp)
    p = ad.patch
    T = tile
    if ad.family == 'hecate':
        if Hp % 10 or Wp % 10:
            T = max(Hm, Wm)            # canvas not a multiple of the 10->9 period: one tile = released behaviour
        else:
            T = max(9 * 64, (tile // 9) * 9)
    t0 = time.time()
    origins = np.array(ad.plan(Hm, Wm, src=src, d=d, order=order, pad=(pr, pc)), np.int64).reshape(-1, 2)
    needm = None
    if need is not None:
        needm = need_in_grid(ad, need, pr, pc, Hm, Wm)
        I = np.pad(needm.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        y0 = np.clip(origins[:, 0], 0, Hm); x0 = np.clip(origins[:, 1], 0, Wm)
        y1 = np.clip(origins[:, 0] + p, 0, Hm); x1 = np.clip(origins[:, 1] + p, 0, Wm)
        origins = origins[(I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) > 0]
    tdir = Path(out) / '.tiles' / f'{ad.name}__{order}__{tag}__{dtag(d)}'
    tdir.mkdir(parents=True, exist_ok=True)
    ntile = npatch = 0
    for ty0 in range(0, Hm, T):
        for tx0 in range(0, Wm, T):
            ty1, tx1 = min(Hm, ty0 + T), min(Wm, tx0 + T)
            tf = tdir / f'{ty0}_{tx0}.npy'
            if tf.exists():
                continue
            if max_tiles is not None and ntile >= max_tiles:
                return None, 'partial'
            sel = origins[(origins[:, 0] < ty1) & (origins[:, 0] + p > ty0) &
                          (origins[:, 1] < tx1) & (origins[:, 1] + p > tx0)] if len(origins) else origins
            res = np.zeros((ty1 - ty0, tx1 - tx0), ad.out_dtype)
            if len(sel):
                by0, bx0 = int(max(0, sel[:, 0].min())), int(max(0, sel[:, 1].min()))
                by1, bx1 = int(min(Hm, sel[:, 0].max() + p)), int(min(Wm, sel[:, 1].max() + p))
                stack = ad.read_grid(src, d, order, (pr, pc), by0, by1, bx0, bx1)
                num = np.zeros((by1 - by0, bx1 - bx0), np.float32); den = np.zeros_like(num)
                kw = dict(Hm=Hm, Wm=Wm) if ad.family == 'hecate' else {}
                npatch += ad.infer(stack, by0, bx0, [tuple(o) for o in sel], num, den, batch=batch, **kw)
                # overlap of the tile with the selected patches' box: with a need mask the box can start inside
                # the tile, so tile-relative slicing must never go negative (bug found 2026-10-11, maps were 0)
                oy0, oy1 = max(ty0, by0), min(ty1, by1); ox0, ox1 = max(tx0, bx0), min(tx1, bx1)
                if oy1 > oy0 and ox1 > ox0:
                    res[oy0 - ty0:oy1 - ty0, ox0 - tx0:ox1 - tx0] = ad.finalize(
                        num[oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0], den[oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0])
            tmp = tf.with_name(tf.name + '.tmp.npy')
            np.save(tmp, res); os.replace(tmp, tf)
            ntile += 1
    full = np.zeros((Hm, Wm), ad.out_dtype)
    for ty0 in range(0, Hm, T):
        for tx0 in range(0, Wm, T):
            r = np.load(tdir / f'{ty0}_{tx0}.npy')
            full[ty0:ty0 + r.shape[0], tx0:tx0 + r.shape[1]] = r
    if needm is not None:
        full[~needm] = 0
    gr, gc = ad.pad_in_grid(pr, pc)
    full = full[gr:, gc:]
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp.with_name(mp.name + '.tmp.tif')
    tifffile.imwrite(tmp, full, compression='zlib')
    os.replace(tmp, mp)
    secs = time.time() - t0
    meta = dict(model=ad.name, family=ad.family, d=int(d), order=order, phase=[pr, pc], grid_scale=ad.grid_scale,
                window=ad.window(d, src.n), shape=list(full.shape), render=str(src.path), n_layers=src.n,
                patches_run=int(npatch), tiles_run=ntile, secs=round(secs, 1), device=str(ad.device),
                need_px=None if need is None else int(need.sum()), layer_perm=getattr(src, 'perm_info', None),
                encoding={'villa': 'uint8 = trunc(255 p) (villa infer)', 'hecate': 'uint8 = rint(255 p), 9.6 um grid',
                          'v8in': 'uint16 = rint(65535 p)'}[ad.family])
    mp.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    for f in tdir.glob('*.npy'):
        f.unlink()
    tdir.rmdir()
    with open(Path(out) / 'timings.tsv', 'a') as f:
        f.write('\t'.join(str(v) for v in (ad.name, d, order, tag, round(secs, 1), src.H * src.W, npatch,
                                            ad.device, time.strftime('%Y-%m-%dT%H:%M:%S'))) + '\n')
    log(f'{mp.name} {full.shape} patches {npatch} {secs:.0f}s')
    return mp, 'done'


def merge_phases(out, model, order, d, tags, how, render_hw, log):
    mp = map_path(out, model, order, 'merged', d)
    if mp.exists():
        return mp
    srcs = [map_path(out, model, order, t, d) for t in tags]
    if not all(s.exists() for s in srcs):
        return None
    ms = [tifffile.imread(s) for s in srcs]
    h = min(m.shape[0] for m in ms); w = min(m.shape[1] for m in ms)
    ms = [m[:h, :w] for m in ms]
    if how == 'mean':
        m = np.rint(np.mean([x.astype(np.float64) for x in ms], 0)).astype(ms[0].dtype)
    elif how.startswith('mask:'):
        mask = np.load(how[5:]).astype(bool)
        if mask.shape != tuple(render_hw):
            raise SystemExit(f'merge mask shape {mask.shape} != render {render_hw}')
        if (h, w) != tuple(render_hw):
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = np.where(mask, ms[1], ms[0])
    else:
        raise SystemExit(f'unknown merge {how}')
    tifffile.imwrite(mp, m, compression='zlib')
    mp.with_suffix('.json').write_text(json.dumps(dict(merged_from=[str(s) for s in srcs], how=how)))
    log(f'merged {mp.name}')
    return mp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render')
    ap.add_argument('--models', help='comma list of ' + ','.join(A.SPECS) + ' or "ensemble4"')
    ap.add_argument('--depths', default='-15..15')
    ap.add_argument('--order', choices=('reverse', 'forward', 'both'), default='reverse')
    ap.add_argument('--phase', action='append', help='PR,PC (repeatable); default 0,0')
    ap.add_argument('--merge', default='none')
    ap.add_argument('--out')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument("--tile", type=int, default=2048, help='work tile (model-grid px) = resume granularity')
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--max-tiles', type=int, help='stop after N new tiles (testing the resume path)')
    ap.add_argument('--status', action='store_true', help='print which models can run and exit')
    ap.add_argument('--need-mask', help='bool (H, W) npy: compute only patches touching these pixels (exact there)')
    ap.add_argument('--shuffle-seed', type=int, help='SABOTAGE: read the rendered layers in a fixed random order')
    a = ap.parse_args(argv)
    if a.status:
        print(json.dumps(A.status(), indent=1)); return 0
    if not (a.render and a.models and a.out):
        ap.error('--render, --models and --out are required')
    gpu_guard(a.device)
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    models = A.ENSEMBLE4 if a.models == 'ensemble4' else a.models.split(',')
    depths = C.centre_out(C.parse_depths(a.depths))
    orders = ['reverse', 'forward'] if a.order == 'both' else [a.order]
    phases = [tuple(int(v) for v in p.split(',')) for p in (a.phase or ['0,0'])]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = A.RenderSource(a.render); src.path = Path(a.render)
    if a.shuffle_seed is not None:
        src.shuffle_layers(a.shuffle_seed)
    need = np.load(a.need_mask).astype(bool) if a.need_mask else None
    if need is not None and need.shape != (src.H, src.W):
        sys.exit(f'need mask {need.shape} != render {(src.H, src.W)}')
    mmask = None
    if a.merge.startswith('mask:'):
        mmask = np.load(a.merge[5:]).astype(bool)
    logf = open(out / 'sweep.log', 'a')

    def log(s):
        line = f'{time.strftime("%H:%M:%S")} {s}'
        print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'render {a.render} {src.n}x{src.H}x{src.W}; models {models}; depths {depths}; orders {orders}; '
        f'phases {phases}; device {a.device}')
    for m in models:                     # fail fast and clearly on missing weights, before any compute
        A.make(m).check()
    rc = 0
    for m in models:
        ad = A.make(m).load(a.device)
        bs = a.batch or ({'villa': 16, 'hecate': 4, 'v8in': 16}[ad.family])
        for d in depths:
            try:
                ad.window(d, src.n)
            except ValueError as e:
                log(f'skip {m} d={d}: {e}'); continue
            for order in orders:
                for pi, ph in enumerate(phases):
                    nd = need
                    if need is not None and mmask is not None and len(phases) == 2:
                        nd = need & (mmask if pi == 1 else ~mmask)    # each pixel only in the phase the merge takes
                    r, st = run_one(ad, src, d, order, ph, out, a.tile, bs, log, a.max_tiles, need=nd)
                    if st == 'partial':
                        log('max-tiles reached; stopping (resume by re-running)'); return 0
                if len(phases) > 1 and a.merge != 'none':
                    merge_phases(out, m, order, d, [f'ph{p[0]}_{p[1]}' for p in phases], a.merge, (src.H, src.W), log)
        del ad
    log('sweep complete')
    return rc


if __name__ == '__main__':
    sys.exit(main())
