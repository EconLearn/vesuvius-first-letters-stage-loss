"""Tile-selection rules vs KNOWN text (the PHerc0343 winner's 18 letters mapped onto our q343x surface).
Same screen maps for every rule (band_screen dense_native, mean of d -2/0/+2). Budget: tiles taken by score until 10 % of
the valid pixels are selected. A letter is caught if >= 30 % of its box is inside selected tiles. Random baseline: 500
random orderings of each rule's eligible tiles at the same budget. Rules fixed before looking at the result:
  A  current kit: 512 px tiles, edge erosion 128 px, >= 50 % valid, >= 25 % of the tile inside the eroded mask, p95 - p50
  B  256 px tiles, edge 32 px, >= 40 % valid, >= 25 % inside, p95 - p50
  C  as B, score on (map - 1 cm median background) to remove the winding-angle texture
usage: eval_select.py WORKDIR REGIONS_JSON"""
import glob, json, sys
from pathlib import Path
import numpy as np, tifffile
from scipy import ndimage
W = Path(sys.argv[1]); R = json.loads(Path(sys.argv[2]).read_text()); plan = json.loads((W / 'plan.json').read_text())
r0, r1 = plan['rows']; c0 = plan['strips'][0]['keep'][0]; c1 = plan['strips'][-1]['keep'][1]
M = np.full((r1 - r0, c1 - c0), np.nan, np.float32); V = np.zeros((r1 - r0, c1 - c0), bool)
for st in plan['strips']:
    sp = W / 'strips' / f"{st['i']:04d}"; fs = sorted(glob.glob(str(sp / 'sw/maps/dnative/dnative__reverse__ph0_0__d*.tif')))
    if len(fs) < 3: print('strip', st['i'], 'missing maps'); continue
    m = np.mean([tifffile.imread(f).astype(np.float32) for f in fs], 0); vz = np.load(sp / 'valid.npz'); v = vz[vz.files[0]].astype(bool)
    ra, rb = st['render']; ka, kb = st['keep']
    assert m.shape == (r1 - r0, rb - ra), (m.shape, (r1 - r0, rb - ra))
    M[:, ka - c0:kb - c0] = m[:, ka - ra:kb - ra]; V[:, ka - c0:kb - c0] = v[:, ka - ra:kb - ra]
M[~V] = np.nan; nvalid = V.sum(); print(f'canvas rows {r0}-{r1} cols {c0}-{c1}; valid {nvalid / 1e6:.1f} Mpx')
small = np.nan_to_num(M[::8, ::8], nan=np.nanmedian(M)); bg = ndimage.median_filter(small, size=int(round(1000 / 8.64 / 8)) | 1)
BG = np.kron(bg, np.ones((8, 8), np.float32))[:M.shape[0], :M.shape[1]]; MC = M - BG
letters = [b for b in R['letters_raw']]; nulls = R['nulls_raw']


def tiles(T, edge, minv):
    er = ndimage.binary_erosion(V, iterations=edge) if edge else V
    out = []
    for y in range(0, M.shape[0] - T + 1, T):
        for x in range((-c0) % T, M.shape[1] - T + 1, T):          # aligned to canvas multiples of T (as the kit)
            v = V[y:y + T, x:x + T]; fv = v.mean()
            if fv < minv or er[y:y + T, x:x + T].mean() < 0.25: continue
            out.append((y, x, int(v.sum())))
    return out


def select(tl, score_map, T, order=None):
    if order is None:
        sc = [np.nanpercentile(score_map[y:y + T, x:x + T], 95) - np.nanpercentile(score_map[y:y + T, x:x + T], 50) for y, x, _ in tl]
        order = np.argsort(sc)[::-1]
    sel = np.zeros(M.shape, bool); tot = 0
    for i in order:
        y, x, n = tl[i]
        if tot >= 0.10 * nvalid: break
        sel[y:y + T, x:x + T] = True; tot += n
    return sel


def caught(sel, boxes):
    hits = []
    for a, b, c, d in boxes:
        ya, yb, xa, xb = a - r0, b - r0, c - c0, d - c0
        if xb <= 0 or xa >= sel.shape[1]: hits.append(None); continue
        box = sel[max(ya, 0):yb, max(xa, 0):xb]; hits.append(bool(box.mean() >= 0.30) if box.size else None)
    return hits


rng = np.random.default_rng(0); res = {}
for name, T, edge, minv, smap in (('A', 512, 128, 0.5, M), ('B', 256, 32, 0.4, M), ('C', 256, 32, 0.4, MC)):
    tl = tiles(T, edge, minv); elig = sum(n for *_, n in tl) / nvalid
    sel = select(tl, smap, T); h = caught(sel, letters)
    hn = [sum(x for x in caught(sel, ns) if x) for ns in nulls] if nulls and isinstance(nulls[0][0], (list, tuple)) else caught(sel, nulls)
    inside = [x for x in h if x is not None]; rec = sum(inside) / max(len(inside), 1)
    rnd = [np.mean([x for x in caught(select(tl, smap, T, rng.permutation(len(tl))), letters) if x is not None]) for _ in range(500)]
    res[name] = dict(tiles=len(tl), eligible_frac=round(float(elig), 3), letters_in_area=len(inside), caught=int(sum(inside)), recall=round(rec, 3),
                     random_mean=round(float(np.mean(rnd)), 3), random_p95=round(float(np.percentile(rnd, 95)), 3),
                     p_random_ge=round(float(np.mean(np.array(rnd) >= rec)), 3), null_sets_caught=hn)
    print(name, json.dumps(res[name]))
(W / 'select_eval.json').write_text(json.dumps(res, indent=1))
