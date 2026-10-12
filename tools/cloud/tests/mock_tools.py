#!/usr/bin/env python
"""Dry-run stand-ins for the heavy tools (CPU, seconds, a few MB). They write the same files, names and formats the
real tools write, so the job scripts and the real helper code (band_screen select/ens drawing, ensemble_maps,
display, stage2_matched -> score_matched, timing_report, cost_guard) run unchanged on top of them.
  render   (render66 / r66.py args)       zarr (66, H, W) with rendered_k attrs; constant-ish CT, invalid corner
  sweep    (depth_sweep args)              uint8 maps + json + timings.tsv; a few 'inky' bright patches
  fit TAG STEPS TS [SPEC]                  fit log with PROGRESS lines (MOCK_FIT_RATE='pre,post' it/s) and 'step N: loss = X'
                                           lines + checkpoint; X settles at a level per winding sense (the SPEC's
                                           spiral_outward_sense; MOCK_LOSS_MAP='tagprefix=CWloss/ACWloss;...')
  flatten OUT DATASET [VOXEL_UM]           small tifxyz (cylindrical sheet around the dataset's umbilicus)
  inputs norm|umb|trk DIR SCROLL Z0 Z1     dataset files
  inputs desc DIR SCROLL Z0 Z1 DS QA_PNG   umbilicus.json + spiral-scroll*.json (voxel size and sense from the config) + a QA image
  stage2 --out W [--rc N]                  stage2_q343.py --run outputs with letter-box ink and a flat sabotage
  gpu OUT.csv                              nvidia-smi samples until killed
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2]                       # tools/: vcpaths.py
sys.path.insert(0, str(TOOLS))
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

ROOT = Path(_P.ROOT)
ENS = Path(os.environ.get('ENS_LANE', str(TOOLS / 'ensemble')))
QUIET = Path(os.environ.get('QUIET_WORK', os.path.join(_P.WORK, 'quiet')))                      # fit supervisor work dir (logs/)
RUNS = Path(os.environ.get('QUIET_RUNS', str(QUIET / 'runs')))                                  # <tag>/checkpoint_fitted.ckpt
sys.path.insert(0, str(ENS))
RNG = np.random.default_rng(0)


def rngs(s):
    a, b = s.split(':'); return int(a), int(b)


def write_render(out, H, W, krange=(0, 66), pixel_mask=None, ink_boxes=()):
    from ens import common as C
    k0, k1 = krange
    vol = np.zeros((66, H, W), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (90 + 20 * np.sin(xx / 37.0) + 10 * np.cos(yy / 23.0)).astype(np.uint8)
    valid = ~((yy < H // 6) & (xx < W // 6))                 # an invalid corner
    if pixel_mask is not None:
        valid &= np.load(pixel_mask).astype(bool)
    for k in range(k0, k1):
        vol[k] = np.where(valid, base, 0)
    C.write_render(out, vol, dict(ensemble_render=dict(n_layers=66, rendered_k=[k0, k1], mock=True, flip_normals=False,
                                                       outward_fraction_plusd=1.0, valid_px=int(valid.sum()))))
    return valid


def cmd_render(argv):
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); ap.add_argument('--rows'); ap.add_argument('--cols')
    ap.add_argument('--k-range', default='0:66'); ap.add_argument('--pixel-mask')
    a, _ = ap.parse_known_args(argv)
    (r0, r1), (c0, c1) = rngs(a.rows), rngs(a.cols)
    write_render(a.out, r1 - r0, c1 - c0, rngs(a.k_range), a.pixel_mask)
    print('mock render', a.out, (66, r1 - r0, c1 - c0))
    return 0


def cmd_sweep(argv):
    import tifffile
    from ens import common as C
    ap = argparse.ArgumentParser(); ap.add_argument('--render'); ap.add_argument('--models'); ap.add_argument('--depths')
    ap.add_argument('--out'); ap.add_argument('--need-mask'); ap.add_argument('--phase', action='append')
    ap.add_argument('--merge', default='none')
    a, _ = ap.parse_known_args(argv)
    arr, at = C.open_render(a.render)
    v = C.valid_mask_all_layers(arr, k_range=at['ensemble_render'].get('rendered_k'))
    if a.need_mask:
        v &= np.load(a.need_mask)
    H, W = v.shape
    out = Path(a.out)
    tags = ['ph0_0'] if not a.phase else [f"ph{p.replace(',', '_')}" for p in a.phase]
    if a.merge != 'none' and len(tags) > 1:
        tags.append('merged')
    ink = np.zeros((H, W), np.float32)
    for (y, x) in [(H // 3, W // 5), (H // 2, W // 2), (2 * H // 3, 4 * W // 5)]:
        ink[max(0, y - 40):y + 40, max(0, x - 60):x + 60] = 1
    for m in a.models.split(','):
        (out / 'maps' / m).mkdir(parents=True, exist_ok=True)
        for d in C.parse_depths(a.depths):
            for tag in tags:
                mp = np.clip(70 + 25 * RNG.standard_normal((H, W)) + 120 * ink * (1 - abs(d) / 8), 0, 255).astype(np.uint8)
                mp[~v] = 0
                f = out / 'maps' / m / f'{m}__reverse__{tag}__d{d:+03d}.tif'
                tifffile.imwrite(f, mp, compression='zlib')
                f.with_suffix('.json').write_text(json.dumps(dict(model=m, d=d, mock=True)))
            with open(out / 'timings.tsv', 'a') as t:
                secs = 4.0 * v.sum() / 1e6 + 0.5
                t.write('\t'.join(map(str, (m, d, 'reverse', 'ph0_0', round(secs, 2), H * W, int(v.sum() / 4096) + 1,
                                            'mock', time.strftime('%Y-%m-%dT%H:%M:%S')))) + '\n')
    print('mock sweep', a.out, a.models, a.depths, tags)
    return 0


def mock_final_loss(tag, spec):
    """Final total loss of a mock fit. Level per sense from MOCK_LOSS_MAP ('tagprefix=CWloss/ACWloss;...', matched against the
    tag without its _cw / _acw suffix); default 270 (CW) / 410 (ACW). The sense is the scroll spec's spiral_outward_sense."""
    sense = json.loads(Path(spec).read_text()).get('spiral_outward_sense') if spec else None
    base = tag.rsplit('_', 1)[0] if tag.endswith(('_cw', '_acw')) else tag
    levels = (270.0, 410.0)
    for item in filter(None, os.environ.get('MOCK_LOSS_MAP', '').split(';')):
        k, v = item.split('=')
        if base.startswith(k):
            levels = tuple(float(x) for x in v.split('/'))
    return levels[1] if sense == 'ACW' else levels[0]


def cmd_fit(argv):
    import torch
    tag, steps, ts = argv[0], int(argv[1]), int(argv[2])
    spec = argv[3] if len(argv) > 3 else None
    final = mock_final_loss(tag, spec)
    pre, post = (float(x) for x in os.environ.get('MOCK_FIT_RATE', '3.4,2.9').split(','))
    logd = QUIET / 'logs'; logd.mkdir(parents=True, exist_ok=True)
    lines = [f'==== START {time.strftime("%Y-%m-%dT%H:%M:%S")} resume=none overrides={{"mock": true}}']
    t = 0.0
    for n in range(0, steps + 1, max(1, steps // 100)):
        r = pre if n <= ts else post
        t += (max(1, steps // 100)) / r
        el = f'{int(t // 60)}m {int(t % 60):02d}s' if t >= 60 else f'{int(t)}s'
        lines.append(f'PROGRESS Optimizing — {n:,}/{steps:,} iterations ({100 * n / steps:.1f}%) — {r:.1f} it/s '
                     f'— elapsed {el} — ETA 0s')
    for n in range(0, steps, 200):   # fit_spiral.py logs the total loss every 200 iterations
        lines.append(f'step {n}: loss = {final * (1 + 9.0 * math.exp(-n / 1200.0)):.1f}, umbilicus = 0.0, track_radius = 0.0, track_dt = 0.0')
    (logd / f'fit_{tag}.log').write_text('\n'.join(lines) + '\n')
    (logd / f'fit_{tag}.super').write_text('mock supervisor exit\n')
    rd = RUNS / tag; rd.mkdir(parents=True, exist_ok=True)
    torch.save({'completed_iterations': steps}, rd / 'checkpoint_fitted.ckpt')
    print('mock fit', tag, steps)
    return 0


def cmd_flatten(argv):
    import tifffile
    out, ds = Path(argv[0]), Path(argv[1])
    cp = json.loads((ds / 'umbilicus.json').read_text())['control_points'][0]
    ux, uy, z0 = cp['x'], cp['y'], cp['z']
    nr, nc, R = 14, 120, 1500.0
    rr, cc = np.mgrid[0:nr, 0:nc].astype(np.float32)
    th = cc * 20.0 / R
    x = ux + R * np.cos(th); y = uy + R * np.sin(th); z = z0 + 100 + rr * 20.0
    x[0, :8] = -1; y[0, :8] = -1; z[0, :8] = -1                 # an invalid corner
    out.mkdir(parents=True, exist_ok=True)
    for n, a in (('x', x), ('y', y), ('z', z)):
        tifffile.imwrite(out / f'{n}.tif', a.astype(np.float32))
    (out / 'meta.json').write_text(json.dumps(dict(format='tifxyz', scale=[0.05, 0.05], area_cm2=0.4, mock=True,
                                                   voxel_um=float(argv[2]) if len(argv) > 2 else 8.64)))
    print('mock flatten', out)
    return 0


def config_row(dataset):
    """The dataset's row of the screening config as a list of columns, else None."""
    p = Path(os.environ.get('SCREEN_CONFIG', str(Path(__file__).resolve().parents[1] / 'screen_scrolls.tsv')))
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split('\t')
            if len(f) > 11 and f[0] == dataset:
                return f
    return None


def config_sense(dataset):
    """z_top_to_bottom column ('true' / 'false' / '-') of the dataset's row in the screening config, else None."""
    f = config_row(dataset)
    return f[11] if f else None


def config_voxel(dataset):
    f = config_row(dataset)
    return float(f[12]) if f and len(f) > 12 and f[12] not in ('', '-') else 8.64


def cmd_snap(argv):
    """snap SRC_TIFXYZ OUT_TIFXYZ : copy (the real step moves vertices onto the m7 sheet prediction)."""
    import shutil
    src, out = argv[0], argv[1]
    shutil.copytree(src, out, dirs_exist_ok=True)
    print('mock snap', out)
    return 0


def cmd_inputs(argv):
    kind, d, s, z0, z1 = argv[0], Path(argv[1]), argv[2], argv[3], argv[4]
    ds_name = argv[5] if len(argv) > 5 else d.name
    sense = config_sense(ds_name)
    d.mkdir(parents=True, exist_ok=True)
    spec = d / 'spiral-scroll.json'
    if kind == 'norm':
        for n in ('nx', 'ny'):
            (d / 'lasagna_inputs' / f'{s}_{n}.ome.zarr' / '2').mkdir(parents=True, exist_ok=True)
    if kind == 'desc' and os.environ.get('MOCK_DESC_FAIL'):
        print('mock: descriptor build failed (MOCK_DESC_FAIL)'); return 2
    if kind in ('umb', 'desc'):   # descriptors: the 343x band umbilicus, or a missing set built from the m7 prediction
        (d / 'umbilicus.json').write_text(json.dumps(dict(control_points=[dict(z=int(z0), x=4300.0, y=4300.0),
                                                                           dict(z=int(z1), x=4310.0, y=4290.0)])))
        doc = {"name": s, "voxel_size_um": config_voxel(ds_name), "z_direction_is_top_to_bottom": {'true': True, '-': None}.get(sense, False),
               "normal_zarr_group": "2",
               "paths": {"normal_x": f"lasagna_inputs/{s}_nx.ome.zarr", "normal_y": f"lasagna_inputs/{s}_ny.ome.zarr"}}
        spec.write_text(json.dumps(doc, indent=1))
        if sense == '-':   # winding sense unknown (as umb_smooth.py writes it): no sense in the spec + the two variants
            for sn in ('CW', 'ACW'):
                (d / f'spiral-scroll.{sn}.json').write_text(json.dumps(dict(doc, spiral_outward_sense=sn), indent=1))
        if kind == 'desc':
            from PIL import Image
            qa = Path(argv[6]); qa.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (16, 16)).save(qa); Image.new('RGB', (16, 16)).save(str(qa).replace('.png', '_core_zoom.png'))
    if kind == 'trk':
        (d / 'tracks').mkdir(exist_ok=True)
        db = d / 'tracks' / f'{s}_m7_band{z0}_{z1}.dbm'
        db.write_bytes(b'mock'); Path(str(db) + '.crossings.npz').write_bytes(b'mock')
    print('mock inputs', kind, d)
    return 0


def cmd_stage2(argv):
    import tifffile
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); ap.add_argument('--rc', type=int, default=0)
    a = ap.parse_args(argv)
    if a.rc:
        print(json.dumps(dict(our_z_range=[9512.0, 9998.0], his_letter_z_range=[9956.0, 10916.0], letters_covered='3/18')))
        print('NO COVERAGE: mock'); return a.rc
    W = Path(a.out); W.mkdir(parents=True, exist_ok=True)
    H, Wd, R0, C0 = 400, 1200, 1000, 50000
    boxes = []
    for row, y in enumerate((120, 250)):
        for k in range(9):
            x = 140 + k * 105
            boxes.append([R0 + y, R0 + y + 50, C0 + x, C0 + x + 60])
    (W / 'regions_ours.json').write_text(json.dumps(dict(letters_raw=boxes, nulls_raw=[[], [], []],
                                                        canvas=dict(rows=[R0, R0 + H], cols=[C0, C0 + Wd]), mock=True)))
    write_render(W / 'q343_letters.zarr', H, Wd)
    from ens import common as C
    arr, at = C.open_render(W / 'q343_letters.zarr')
    v = C.valid_mask_all_layers(arr)
    lm = np.zeros((H, Wd), np.float32)
    for r0, r1, c0, c1 in boxes:
        lm[r0 - R0 + 10:r1 - R0 - 10, c0 - C0 + 10:c1 - C0 - 10] = 1      # ink strokes inside the boxes
    for sub, tag, depths, inky in (('sweep', 'merged', range(-6, 7, 2), True), ('sweep_shuffled', 'ph0_0', range(-2, 3, 2), False)):
        for m in ('ink9_s42', 'ink9_s43', 'dnative'):
            (W / sub / 'maps' / m).mkdir(parents=True, exist_ok=True)
            for d in depths:
                mp = 70 + 25 * RNG.standard_normal((H, Wd)) + (90 * lm if inky else 0)
                mp = np.clip(mp, 0, 255).astype(np.uint8); mp[~v] = 0
                tifffile.imwrite(W / sub / 'maps' / m / f'{m}__reverse__{tag}__d{d:+03d}.tif', mp, compression='zlib')
    r = subprocess.run([sys.executable, str(ENS / 'ensemble_maps.py'), '--render', str(W / 'q343_letters.zarr'),
                        '--maps', str(W / 'sweep'), '--models', 'ink9_s42,ink9_s43,dnative', '--tag', 'merged',
                        '--stat-depths=-6..6:2', '--write-depths=-6..6:2', '--combine', 'mean', '--out', str(W / 'ens'),
                        '--name', 'q343', '--no-png'])
    print(json.dumps(dict(our_z_range=[9512.0, 10990.0], his_letter_z_range=[9956.0, 10916.0], letters_covered='18/18')))
    return r.returncode


def cmd_gpu(argv):
    with open(argv[0], 'a') as f:
        for i in range(100000):
            f.write(f'2000/01/01 00:00:{i % 60:02d}.000, {55 + (i % 7)}, {6000 + 10 * (i % 5)}\n'); f.flush()
            time.sleep(0.5)
    return 0


if __name__ == '__main__':
    cmd, argv = sys.argv[1], sys.argv[2:]
    sys.exit(dict(render=cmd_render, sweep=cmd_sweep, fit=cmd_fit, flatten=cmd_flatten, snap=cmd_snap, inputs=cmd_inputs,
                  stage2=cmd_stage2, gpu=cmd_gpu)[cmd](argv))
