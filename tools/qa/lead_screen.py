"""lead_screen.py OUT_JSON K SCROLL[,SCROLL...] [--inject]
Model-free screen for lead-bearing ink (Seiler et al., PLOS ONE 2026, e0353485: lead absorbs up to ~25x more X-rays
than carbonized papyrus, detectable at 25 ug/cm2 => letters should be BRIGHT on the sheet surface in raw CT).
Per scroll: K random 128^3 CT blocks on papyrus (picked from the published m7 sheet mask at a coarse level), streamed
from public S3 into memory (nothing written). Statistics:
  - E99.9 = P(near-sheet | CT >= scroll p99.9) / P(near-sheet | inside scroll)  (bright voxels enriched on sheets?)
  - flat bright components on the sheet: 3D components of (CT >= p99.9) & near-sheet with >= 40 voxels whose
    PCA thickness ratio sqrt(l3/l1) < 0.25 (thin, sheet-like, as an ink layer would be), per 10^6 sheet voxels
  - saturation: fraction of near-sheet voxels at >= 250
--inject: POSITIVE CONTROL. Before scoring, add +LEAD_INJECT_DELTA (env, default 100; clipped at 255) to a striped 1-voxel layer on the sheet surface in
every block (a crude lead-ink stand-in, ~10 % of surface area). The statistics must jump, or the screen is blind."""
import sys, json, time
import numpy as np
import s3fs, zarr
from scipy import ndimage
from pathlib import Path

B = 'vesuvius-challenge-open-data'
fs = s3fs.S3FileSystem(anon=True)
store_opts = dict(anon=True)
import os
INJ_DELTA = int(os.environ.get('LEAD_INJECT_DELTA', '100'))   # grey levels added by --inject (control strength)


def find_paths(scroll):
    vols = sorted(p.split('/')[-1] for p in fs.ls(f'{B}/{scroll}/volumes/') if p.endswith('masked.zarr'))
    if not vols:
        vols = sorted(p.split('/')[-1] for p in fs.ls(f'{B}/{scroll}/volumes/') if p.endswith('.zarr'))
    vol = vols[0]
    stamp = vol.split('-')[0]
    names = [p.split('/')[-1] for p in fs.ls(f'{B}/{scroll}/representations/predictions/surfaces/')]
    m7 = sorted(n for n in names if n.endswith('-surface-m7-L0-th0.2.zarr') and n.startswith(stamp))
    if not m7:
        m7 = sorted(n for n in names if n.endswith('-surface-m7-L0-th0.2.zarr'))
    return vol, (m7[-1] if m7 else None)


def open_arr(path):
    return zarr.open(f's3://{path}', mode='r', storage_options=store_opts)


def pick_blocks(m7g, K, rng):
    lv = None
    for L in ('4', '3', '5'):
        try:
            a = m7g[L]; lv = int(L); break
        except Exception:
            continue
    f = 2 ** lv
    Z = a.shape[0]
    cands = []
    for zf in (0.35, 0.5, 0.65):
        zi = int(Z * zf)
        sl = np.asarray(a[zi])
        yy, xx = np.nonzero(sl > 0)
        if len(yy):
            sel = rng.choice(len(yy), size=min(400, len(yy)), replace=False)
            cands += [(zi * f, yy[i] * f, xx[i] * f) for i in sel]
    rng.shuffle(cands)
    seen, out = set(), []
    for c in cands:
        k = tuple(int(v) // 128 for v in c)
        if k not in seen:
            seen.add(k); out.append(k)
        if len(out) >= 3 * K:
            break
    return out


def thin_components(mask, minvox=40):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    flat = big = 0
    for i in np.nonzero(sizes >= minvox)[0]:
        pts = np.argwhere(lab == i + 1).astype(np.float64)
        ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))
        big += 1
        if ev[2] > 0 and np.sqrt(max(ev[0], 0) / ev[2]) < 0.25:
            flat += 1
    return flat, big


def scan(scroll, K, inject, seed=0):
    t0 = time.time()
    vol, m7n = find_paths(scroll)
    if m7n is None:
        return dict(scroll=scroll, error='no m7 prediction')
    ct = open_arr(f'{B}/{scroll}/volumes/{vol}')['0']
    m7g = open_arr(f'{B}/{scroll}/representations/predictions/surfaces/{m7n}')
    m70 = m7g['0']
    rng = np.random.default_rng(seed)
    blocks = pick_blocks(m7g, K, rng)
    vals_inside, vals_sheet, used = [], [], []
    data = []
    for k in blocks:
        if len(used) >= K:
            break
        z0, y0, x0 = (v * 128 for v in k)
        if z0 + 128 > ct.shape[0] or y0 + 128 > ct.shape[1] or x0 + 128 > ct.shape[2]:
            continue
        c = np.asarray(ct[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128])
        s = np.asarray(m70[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128]) > 0
        inside = c > 0
        if inside.mean() < 0.9 or s.mean() < 0.05:
            continue
        if inject:
            surf = s & ~ndimage.binary_erosion(s)
            zz, yy, xx = np.indices(s.shape)
            stripe = ((yy // 6) % 5 == 0) | ((xx // 6) % 5 == 0)
            add = surf & stripe & (rng.random(s.shape) < 0.5)
            c = np.clip(c.astype(np.int16) + INJ_DELTA * add, 0, 255).astype(np.uint8)
        near = ndimage.binary_dilation(s, iterations=2)
        data.append((c, near, inside, s))
        used.append(k)
    if not data:
        return dict(scroll=scroll, vol=vol, error='no usable blocks')
    allin = np.concatenate([c[i] for c, n, i, s in data])
    T = float(np.percentile(allin, 99.9))
    n_br = n_br_near = n_in = n_near = n_sheet = sat_near = 0
    flat = big = 0
    for c, near, inside, s in data:
        br = (c >= T) & inside
        n_br += br.sum(); n_br_near += (br & near).sum()
        n_in += inside.sum(); n_near += (near & inside).sum(); n_sheet += s.sum()
        sat_near += ((c >= 250) & near).sum()
        f, b = thin_components(br & near)
        flat += f; big += b
    E = (n_br_near / max(n_br, 1)) / (n_near / max(n_in, 1))
    return dict(scroll=scroll, vol=vol, m7=m7n, blocks=len(data), T_p99_9=T,
                median_inside=float(np.median(allin)), p99_inside=float(np.percentile(allin, 99)),
                E_bright_on_sheet=round(float(E), 3), near_sheet_frac=round(float(n_near / n_in), 3),
                flat_bright_per_Msheet=round(1e6 * flat / max(n_sheet, 1), 2), bright_comps=int(big), flat_comps=int(flat),
                sat_frac_near=float(sat_near / max(n_near, 1)), secs=round(time.time() - t0, 1), inject=inject,
                inject_delta=INJ_DELTA if inject else 0)


if __name__ == '__main__':
    out, K, scrolls = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
    inject = '--inject' in sys.argv
    res = []
    for s in scrolls:
        try:
            r = scan(s, K, inject)
        except Exception as e:
            r = dict(scroll=s, error=repr(e)[:300])
        res.append(r); print(json.dumps(r), flush=True)
        Path(out).write_text(json.dumps(res, indent=1))
