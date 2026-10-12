#!/usr/bin/env python
"""calib343_regions: letter boxes, null boxes and the tile-phase merge mask for the PHerc 343 calibration, from his
released files ONLY (geometry; no ink map is read, so the nulls are fixed before any of our ink is computed).

Input  --rel DIR with (his release):
         concat_w047-w048_R5B2_z9500-11000.json              sidecar (letters, area, orientation, render shape)
         concat_w047-w048_R5B2_z9500-11000.tifxyz/           joined mesh (one grid step = 20 voxels)
         w047_R5B2_z9500-11000.tifxyz/                       source mesh (for the merge mask)
         concat_w047-w048_R5B2_z9500-11000.png               optional: his image, used only to decide the frame of
                                                             the letter boxes (yellow inside boxes vs outside)
Letter-box frame: his letter_sizes_csv note says "area pixels (reading orientation, origin at the top left of the
area)"; the 'letters' block says frame 'raw'. Both hypotheses are tested against his PNG when given (default:
area_reading). Boxes are converted to RAW render pixels of his full 66-layer canvas (shape from the sidecar).
Nulls: for every letter box, 3 partner boxes of the same size at the nearest positions (rings of 40 px up to 3200 px,
32 directions) that are inside the canvas, >= 90 % on valid mesh, overlap no letter box (dilated 40 px), no other box
of the same null set, and none of the letter's own other partners. Null set k = the k-th partners of all boxes.
(Rule revised 2026-10-10 before any 343 ink was computed: the first version also forbade overlap across null sets
and left row-2 letters without partners.)
Merge mask: render pixels whose mesh node lies on w047 (nearest w047 vertex < 3 vox) -> taken from the padded
(40, 200) phase run, as his merge_tile_phase.py does (w047 nodes incl. the transition strip).
Output: --out regions.json (+ merge_w047.npy next to it).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens import common as C  # noqa: E402

NAME = 'concat_w047-w048_R5B2_z9500-11000'


def find(js, key):
    if isinstance(js, dict):
        if key in js:
            return js[key]
        for v in js.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(js, list):
        for v in js:
            r = find(v, key)
            if r is not None:
                return r
    return None


def boxes_to_raw(boxes, frame, area, H, W, flip_ud, flip_lr):
    """[x0, y0, x1, y1] (x1/y1 exclusive) -> raw render [r0, r1, c0, c1] ints (outward rounding)."""
    out = []
    for x0, y0, x1, y1 in boxes:
        if frame == 'raw_full':
            c0, c1, r0, r1 = x0, x1, y0, y1
        else:                                  # area pixels, reading orientation
            if flip_lr:
                c0, c1 = area['col1'] - x1, area['col1'] - x0
            else:
                c0, c1 = area['col0'] + x0, area['col0'] + x1
            if flip_ud:
                r0, r1 = area['row1'] - y1, area['row1'] - y0
            else:
                r0, r1 = area['row0'] + y0, area['row0'] + y1
        out.append([int(np.floor(r0)), int(np.ceil(r1)), int(np.floor(c0)), int(np.ceil(c1))])
    return out


def frame_check(png, js, boxes, area, H, W, flip_ud, flip_lr):
    """Fraction of his yellow overlay inside the boxes, for each frame hypothesis (data panel of his PNG)."""
    import cv2
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    dp = find(js, 'data_panel') or {}
    x, y, w, h = dp.get('x', 0), dp.get('y', 0), dp.get('width', img.shape[1]), dp.get('height', img.shape[0])
    pan = img[y:y + h, x:x + w].astype(np.int16)
    b, g, r = pan[..., 0], pan[..., 1], pan[..., 2]
    yellow = (r > 150) & (g > 130) & (r - b > 60) & (g - b > 40)
    res = {}
    for fr in ('area_reading', 'raw_full'):
        m = np.zeros(yellow.shape, bool)
        for bx in boxes_to_raw(boxes, fr, area, H, W, flip_ud, flip_lr):
            r0, r1, c0, c1 = bx
            # raw -> panel (reading) coords
            if flip_ud:
                py0, py1 = area['row1'] - r1, area['row1'] - r0
            else:
                py0, py1 = r0 - area['row0'], r1 - area['row0']
            if flip_lr:
                px0, px1 = area['col1'] - c1, area['col1'] - c0
            else:
                px0, px1 = c0 - area['col0'], c1 - area['col0']
            m[max(0, py0):max(0, py1), max(0, px0):max(0, px1)] = True
        inside = float(yellow[m].mean()) if m.any() else 0.0
        outside = float(yellow[~m].mean()) if (~m).any() else 0.0
        res[fr] = dict(yellow_inside=inside, yellow_outside=outside, ratio=inside / max(outside, 1e-6))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rel', required=True)
    ap.add_argument('--json'); ap.add_argument('--png')
    ap.add_argument('--frame', choices=('auto', 'area_reading', 'raw_full'), default='auto')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-null', type=int, default=3)
    a = ap.parse_args(argv)
    rel = Path(a.rel)
    jp = Path(a.json) if a.json else C.rel_file(rel, f'{NAME}.json')
    js = json.loads(jp.read_text())
    shape = find(js, 'ct')['shape'] if find(js, 'ct') else None
    n, H, W = shape
    area = find(js, 'area')['raw_render_px']
    ori = find(js, 'orientation')
    flip_ud, flip_lr = bool(ori['flip_ud']), bool(ori['flip_lr'])
    letters = find(js, 'letters')
    if isinstance(letters, dict) and 'letters_area_px' in letters:      # his sidecar: area px, reading orientation
        items = letters['letters_area_px']
    elif isinstance(letters, dict) and 'items' in letters:
        items = letters['items']
    else:
        items = letters
    if isinstance(items, dict):
        items = [v for v in items.values() if isinstance(v, dict) and 'box' in v]
    boxes = [it['box'] for it in items]
    if not boxes:
        sys.exit(f'no letter boxes found in {jp}')
    png = Path(a.png) if a.png else C.rel_file(rel, f'{NAME}.png')
    fc = None
    frame = a.frame
    if frame == 'auto':
        if png.exists():
            fc = frame_check(png, js, boxes, area, H, W, flip_ud, flip_lr)
            frame = max(fc, key=lambda k: fc[k]['ratio'])
        else:
            frame = 'area_reading'
    raw = boxes_to_raw(boxes, frame, area, H, W, flip_ud, flip_lr)
    # cross-check with his inputs/<name>_letter_boxes.json (frame 'raw' = raw render px), when present
    lb = C.rel_file(rel, f'{NAME}_letter_boxes.json')
    xcheck = None
    if lb.exists():
        lbj = json.loads(lb.read_text())
        rb = [[b[1], b[3], b[0], b[2]] for b in (it['box'] for it in lbj['letters'])]   # x0,y0,x1,y1 -> r0,r1,c0,c1
        if len(rb) == len(raw):
            dmax = float(np.max(np.abs(np.array(rb, float) - np.array(raw, float))))
            xcheck = dict(file=str(lb), max_abs_px=dmax, agree=dmax <= 1.0,
                          rule='ground truth = his inputs letter_boxes.json (raw render px); sidecar boxes converted '
                               'from area/reading px must agree within 1 px')
            if dmax > 1.0:
                print(f'WARNING: sidecar boxes ({frame}) disagree with {lb.name} by {dmax:.1f} px')
            raw = [[int(np.floor(r0)), int(np.ceil(r1)), int(np.floor(c0)), int(np.ceil(c1))]
                   for r0, r1, c0, c1 in rb]
    # mesh validity on the render grid
    L = C.use_volume('PHerc0343')
    P, valid_nodes, meta = L.load_tifxyz(C.rel_file(rel, f'{NAME}.tifxyz'))
    gs = float(meta['scale'][0]); step = int(round(1 / gs))
    vmask = np.repeat(np.repeat(valid_nodes, step, 0), step, 1)[:H, :W]
    letter = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in raw:
        letter[r0:r1, c0:c1] = True
    from scipy import ndimage
    forbid = ndimage.binary_dilation(letter, iterations=40)
    integ = np.pad(vmask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    taken = [forbid.copy() for _ in range(a.n_null)]      # per null set: letters (dilated) + that set's boxes
    nulls = [[] for _ in range(a.n_null)]
    dirs = [(np.sin(t), np.cos(t)) for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)]
    for bi, (r0, r1, c0, c1) in enumerate(raw):
        h, w = r1 - r0, c1 - c0
        got = 0
        mine = np.zeros((H, W), bool)                    # this letter's earlier partners (kept distinct)
        for rad in range(40, 3201, 40):
            for sy, sx in dirs:
                nr0, nc0 = int(round(r0 + sy * rad)), int(round(c0 + sx * rad))
                nr1, nc1 = nr0 + h, nc0 + w
                if nr0 < 0 or nc0 < 0 or nr1 > H or nc1 > W:
                    continue
                vf = (integ[nr1, nc1] - integ[nr0, nc1] - integ[nr1, nc0] + integ[nr0, nc0]) / (h * w)
                if vf < 0.9 or taken[got][nr0:nr1, nc0:nc1].any() or mine[nr0:nr1, nc0:nc1].any():
                    continue
                nulls[got].append([nr0, nr1, nc0, nc1])
                taken[got][nr0:nr1, nc0:nc1] = True
                mine[nr0:nr1, nc0:nc1] = True
                got += 1
                if got == a.n_null:
                    break
            if got == a.n_null:
                break
        if got < a.n_null:
            print(f'warning: letter {bi + 1} has only {got} null partners within 3200 px')
    # merge mask: w047 nodes
    from scipy.spatial import cKDTree
    P47, v47, _ = L.load_tifxyz(C.rel_file(rel, 'w047_R5B2_z9500-11000.tifxyz'))
    T = cKDTree(P47[v47])
    dist, _ = T.query(P[valid_nodes], distance_upper_bound=3.0)
    node47 = np.zeros(valid_nodes.shape, bool); node47[valid_nodes] = np.isfinite(dist)
    m47 = np.repeat(np.repeat(node47, step, 0), step, 1)[:H, :W]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_name('merge_w047.npy'), m47)
    reg = dict(render_shape=[n, H, W], canvas=dict(rows=[0, H], cols=[0, W]), frame=frame, frame_check=fc,
               letter_boxes_crosscheck=xcheck,
               flip_ud=flip_ud, flip_lr=flip_lr, area_raw=area, letters_raw=raw,
               letter_labels=[it.get('label', str(i + 1)) for i, it in enumerate(items)], nulls_raw=nulls,
               null_rule='per-letter partner boxes, rings of 40 px (<= 3200) x 32 directions, >= 90 % valid mesh, no '
                         'overlap with letters dilated 40 px, nor within a null set, nor with the same letter\'s '
                         'other partners; set k = k-th partner of every letter',
               merge_mask=str(out.with_name('merge_w047.npy')), merge_w047_fraction=float(m47[vmask].mean()),
               mesh_valid_fraction=float(vmask.mean()), sidecar=str(jp))
    out.write_text(json.dumps(reg, indent=1))
    print(json.dumps(dict(frame=frame, frame_check=fc, crosscheck=xcheck, letters=len(raw), nulls=[len(x) for x in nulls],
                          merge_w047_fraction=reg['merge_w047_fraction'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
