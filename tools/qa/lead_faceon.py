"""lead_faceon.py SCREEN_JSON OUTDIR K SEED SCROLL[,SCROLL...]
Replication + look for the lead-ink outliers: K fresh blocks (new seed), per-block counts of thin bright on-sheet
components (same definition as lead_screen.py, threshold = the scroll's p99.9 from SCREEN_JSON, written by
lead_screen.py), and a face-on
view of the largest thin components: CT max over +-2 vx along the component's normal, on a 97 x 97 vx plane through
its centroid (axes = its two largest PCA directions). Lead letters would show bright strokes there."""
import sys, json
import numpy as np
from pathlib import Path
from scipy import ndimage
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead_screen as LS

screen, out, K, seed, scrolls = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(',')
out.mkdir(parents=True, exist_ok=True)
prev = {r['scroll']: r for r in json.load(open(screen)) if 'T_p99_9' in r}
summary = []
for S in scrolls:
    vol, m7n = LS.find_paths(S)
    ct = LS.open_arr(f'{LS.B}/{S}/volumes/{vol}')['0']
    m7g = LS.open_arr(f'{LS.B}/{S}/representations/predictions/surfaces/{m7n}')
    rng = np.random.default_rng(seed)
    blocks = LS.pick_blocks(m7g, K, rng)
    T = prev[S]['T_p99_9']
    per_block, views, n_sheet, used = [], [], 0, 0
    for k in blocks:
        if used >= K:
            break
        z0, y0, x0 = (v * 128 for v in k)
        if z0 + 128 > ct.shape[0] or y0 + 128 > ct.shape[1] or x0 + 128 > ct.shape[2]:
            continue
        c = np.asarray(ct[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128])
        s = np.asarray(m7g['0'][z0:z0 + 128, y0:y0 + 128, x0:x0 + 128]) > 0
        if (c > 0).mean() < 0.9 or s.mean() < 0.05:
            continue
        used += 1; n_sheet += s.sum()
        near = ndimage.binary_dilation(s, iterations=2)
        m = (c >= T) & (c > 0) & near
        lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
        flat = 0
        if n:
            sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
            for i in np.nonzero(sizes >= 40)[0]:
                pts = np.argwhere(lab == i + 1).astype(np.float64)
                ev, evec = np.linalg.eigh(np.cov(pts.T))
                if ev[2] > 0 and np.sqrt(max(ev[0], 0) / ev[2]) < 0.25:
                    flat += 1
                    views.append((float(sizes[i]), k, pts.mean(0), evec))
        per_block.append(flat)
    # face-on views of the largest thin components (up to 12)
    views.sort(key=lambda v: -v[0])
    tiles = []
    for size, k, cen, evec in views[:12]:
        z0, y0, x0 = (v * 128 for v in k)
        u, w, nrm = evec[:, 2], evec[:, 1], evec[:, 0]
        a = np.arange(-48, 49)
        P = (np.array([z0, y0, x0], float) + cen)[None, None, :] + a[:, None, None] * u + a[None, :, None] * w
        lo = np.floor(P.reshape(-1, 3).min(0) - 4).astype(int); hi = np.ceil(P.reshape(-1, 3).max(0) + 4).astype(int)
        box = np.asarray(ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32)
        layers = [ndimage.map_coordinates(box, (P + t * nrm - lo).reshape(-1, 3).T, order=1).reshape(97, 97)
                  for t in range(-2, 3)]
        mip = np.max(layers, 0)
        img = Image.fromarray(np.clip(mip, 0, 255).astype(np.uint8)).resize((194, 194), Image.NEAREST)
        tiles.append((img, f'{int(size)}vx'))
    if tiles:
        sheet = Image.new('L', (6 * 200, ((len(tiles) + 5) // 6) * 214), 30)
        d = ImageDraw.Draw(sheet)
        for i, (im, lab_) in enumerate(tiles):
            x, y = (i % 6) * 200, (i // 6) * 214
            sheet.paste(im, (x + 3, y + 17)); d.text((x + 4, y + 2), f'{S} {lab_}', fill=255)
        sheet.save(out / f'{S}_faceon.png')
    pb = np.array(per_block)
    r = dict(scroll=S, seed=seed, blocks=int(used), T=T, flat_per_Msheet=round(1e6 * pb.sum() / max(n_sheet, 1), 2),
             flat_total=int(pb.sum()), blocks_with_flat=int((pb > 0).sum()), max_in_one_block=int(pb.max()) if len(pb) else 0,
             top3_block_share=round(float(np.sort(pb)[-3:].sum() / max(pb.sum(), 1)), 2))
    summary.append(r); print(json.dumps(r), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
