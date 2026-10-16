"""Pixel AUC on PHerc1447 w058 / w060 ink labels (inside mask.png), z-scored depth-mean maps (d -4,-2,0,+2), per model and
ensembles; block bootstrap (512 px blocks) 95 % CIs on differences. Pre-registered (LEDGER 2026-10-14):
H1 dense_native alone > equal-weight 3-model mean; H2 4-model (+readerv2) > 3-model. 1447 is in no member's training set.
usage: w1447_score.py HF_SURFACES_DIR WORK_DIR   (WORK_DIR holds the depth-sweep maps sw_<w>/maps/...; tools/ensemble on PYTHONPATH)"""
import sys, json, numpy as np, tifffile
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'ensemble'))
from ens import adapters as A
H, O = Path(sys.argv[1]), Path(sys.argv[2]); DEPTHS = [-4, -2, 0, 2]
MODELS = ['ink9_s42', 'ink9_s43', 'dnative', 'readerv2']
def auc(s, y):
    o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
res = {}; blocks = {}
for w in ('w058', 'w060'):
    lab = np.array(Image.open(H / w / 'labels/inklabels.png').convert('L')) > 127; msk = np.array(Image.open(H / w / 'labels/mask.png').convert('L')) > 127
    Z = {}
    for m in MODELS:
        vals = [A.to_unit(m, tifffile.imread(O / f'sw_{w}' / 'maps' / m / f'{m}__reverse__ph0_0__d{d:+03d}.tif')).astype(np.float32) for d in DEPTHS]
        sh = vals[0].shape; ok = msk[:sh[0], :sh[1]]
        allv = np.concatenate([v[ok] for v in vals]); mu, sd = allv.mean(), allv.std() + 1e-6
        Z[m] = np.mean([(v - mu) / sd for v in vals], 0)
    sh = Z[MODELS[0]].shape; ok = msk[:sh[0], :sh[1]]; y = lab[:sh[0], :sh[1]]
    combos = {**{m: Z[m] for m in MODELS}, 'base3': (Z['ink9_s42'] + Z['ink9_s43'] + Z['dnative']) / 3,
              'plus4': sum(Z.values()) / 4, 'dn_weighted': (Z['ink9_s42'] + Z['ink9_s43'] + 2 * Z['dnative']) / 4}
    by = np.arange(sh[0])[:, None] // 512 * 1000 + np.arange(sh[1])[None, :] // 512; bk = by[ok]; yy = y[ok]
    res[w] = {k: round(float(auc(v[ok], yy)), 4) for k, v in combos.items()}; blocks[w] = (bk, yy, {k: v[ok] for k, v in combos.items()})
    print(w, 'ink px', int(yy.sum()), 'of', int(ok.sum()), json.dumps(res[w]), flush=True)
rng = np.random.default_rng(0)
def diff_ci(a, b, nb=300):
    ds = []
    for _ in range(nb):
        da = []; 
        for w, (bk, yy, cv) in blocks.items():
            ub = np.unique(bk); pick = rng.choice(ub, len(ub)); idx = np.concatenate([np.nonzero(bk == u)[0] for u in pick])
            if yy[idx].sum() == 0 or yy[idx].sum() == len(idx): continue
            da.append(auc(cv[a][idx], yy[idx]) - auc(cv[b][idx], yy[idx]))
        ds.append(np.mean(da))
    return [round(float(x), 4) for x in np.percentile(ds, [2.5, 97.5])]
pt = lambda a, b: round(float(np.mean([res[w][a] - res[w][b] for w in res])), 4)
out = dict(per_surface=res, H1_dnative_minus_base3=dict(mean=pt('dnative', 'base3'), ci95=diff_ci('dnative', 'base3')),
           H2_plus4_minus_base3=dict(mean=pt('plus4', 'base3'), ci95=diff_ci('plus4', 'base3')),
           readerv2_minus_dnative=dict(mean=pt('readerv2', 'dnative'), ci95=diff_ci('readerv2', 'dnative')))
print(json.dumps(out, indent=1)); (O / 'w1447_auc.json').write_text(json.dumps(out, indent=1))
