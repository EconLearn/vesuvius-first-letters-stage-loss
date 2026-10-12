#!/usr/bin/env python
"""display: his make_submission_image.py 'dimmed-ct' style -- ink in yellow over a dimmed CT layer at depth d.

  ink_u8 = p1..p99.5 stretch of the ink map (ensemble z-mean by default, stretch from <name>__stats.json);
  u = ink_u8 / 255; alpha = 0.9 * clip((u - 0.35) / (0.75 - 0.35), 0, 1);  colour RGB (255, 235, 120);
  CT grey (layer 33 + d, stretched p1..p99.5 over layers 18..48) * 0.45, alpha-composited; invalid pixels black;
  1 cm scale bar = 10000 / voxel_um px (1157 px at 8.64 um).
Plus line-scale strips (--strip-h px tall, 25 % overlap, optional --zoom) so an observer can judge letters at the
scale of one text line, and an optional --crop r0:r1,c0:c1 and reading-orientation flips (--flip-ud/--flip-lr).

usage: display.py --render R.zarr --ens-dir DIR --name NAME [--ink-d 0 | --ink dmax] [--ct-d 1] --out PNG
                  [--strips DIR --strip-h 450 --zoom 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens import common as C  # noqa: E402

COL_BGR = np.array([120, 235, 255], np.float32)   # RGB (255, 235, 120)
ALPHA, THR, FULL, CT_DIM = 0.9, 0.35, 0.75, 0.45


def compose(ct_u8, ink_u8, valid):
    u = ink_u8.astype(np.float32) / 255.
    al = ALPHA * np.clip((u - THR) / max(FULL - THR, 1e-6), 0, 1)
    img = np.repeat(ct_u8.astype(np.float32)[..., None], 3, -1) * np.float32(CT_DIM)
    img = img * (1 - al[..., None]) + COL_BGR * al[..., None]
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    img[~valid] = 0
    return img


def scale_bar(img, voxel_um, label='1 cm'):
    import cv2
    bar = int(round(10000.0 / voxel_um))
    h, w = img.shape[:2]
    if bar > w - 20:
        bar = int(round(1000.0 / voxel_um)); label = '1 mm'
    x0, y0 = 10, h - 20
    cv2.rectangle(img, (x0, y0 - 5), (x0 + bar, y0 + 5), (255, 255, 255), -1)
    cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render', required=True)
    ap.add_argument('--ens-dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--ink', default=None, help="ensemble tag: 'd+01', 'dmax', 'dmean' (default d of --ink-d)")
    ap.add_argument('--ink-d', type=int, default=1)
    ap.add_argument('--ink-npy', help='any float map (H, W) instead of the ensemble; stretched p1..p99.5 of valid')
    ap.add_argument('--ct-d', type=int, default=1, help='CT layer depth (his image: d = +1)')
    ap.add_argument('--voxel-um', type=float, default=8.64)
    ap.add_argument('--crop')
    ap.add_argument('--flip-ud', action='store_true'); ap.add_argument('--flip-lr', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--strips'); ap.add_argument('--strip-h', type=int, default=450)
    ap.add_argument('--zoom', type=float, default=1.0)
    a = ap.parse_args(argv)
    arr, _ = C.open_render(a.render)
    n = arr.shape[0]; mid = n // 2
    ed = Path(a.ens_dir)
    st = json.loads((ed / f'{a.name}__stats.json').read_text())
    valid = np.load(ed / f'{a.name}__valid.npy')
    if a.ink_npy:
        X = np.load(a.ink_npy).astype(np.float32)
        lo, hi = np.percentile(X[valid], [1, 99.5])
    else:
        tag = a.ink or f'd{a.ink_d:+03d}'
        X = np.load(ed / f'{a.name}__ensemble__{tag}.npy')
        key = 'ensemble' if tag.startswith('d+') or tag.startswith('d-') else f'ensemble_{tag}'
        lo, hi = st[key]['p1'], st[key]['p99_5']
    ink = (np.clip((X - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    ink[~valid] = 0
    _, rat = C.open_render(a.render)
    rk = rat.get('ensemble_render', {}).get('rendered_k', [0, n])
    ctvalid = C.valid_mask_all_layers(arr, k_range=rk)
    ctl = np.asarray(arr[mid + a.ct_d]).astype(np.float32)
    ct = (np.clip((ctl - st['ct']['p1']) / max(st['ct']['p99_5'] - st['ct']['p1'], 1e-9), 0, 1) * 255).astype(np.uint8)
    ink[~valid] = 0
    img = compose(ct, ink, ctvalid)        # CT shown wherever rendered; ink only on the ensemble's valid pixels
    if a.crop:
        (r0, r1), (c0, c1) = [tuple(int(v) for v in s.split(':')) for s in a.crop.split(',')]
        img = img[r0:r1, c0:c1]
    if a.flip_ud:
        img = img[::-1]
    if a.flip_lr:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img)
    full = scale_bar(img.copy(), a.voxel_um)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, full)
    print('wrote', a.out, full.shape)
    if a.strips:
        sd = Path(a.strips); sd.mkdir(parents=True, exist_ok=True)
        h = a.strip_h; step = max(1, int(h * 0.75)); k = 0
        for r0 in range(0, max(1, img.shape[0] - h // 4), step):
            s = img[r0:r0 + h]
            if a.zoom != 1:
                s = cv2.resize(s, None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_NEAREST)
            s = scale_bar(np.ascontiguousarray(s), a.voxel_um / a.zoom)
            cv2.imwrite(str(sd / f'strip_{k:02d}_r{r0}.png'), s); k += 1
        print('strips', k, 'in', sd)
    return 0


if __name__ == '__main__':
    sys.exit(main())
