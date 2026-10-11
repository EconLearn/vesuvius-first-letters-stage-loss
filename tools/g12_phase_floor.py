import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
from vlib import metrics as M
out = {}
pooled = {}
for w in G.WINDS:
    g = np.load(G.W / 'org' / w / 'geo.npz'); band = g['band']; lab = g['label'].astype(bool) & band
    preds = {'R0_0': G.load_pair(f'{w}_R')}
    for dr, dc in G.PHASES:
        f, r = G.load_pair(f'{w}_Rph{dr}_{dc}'); preds[f'R{dr}_{dc}'] = (f[dr:, dc:], r[dr:, dc:])
    aucs = {k: M.auc(v[1], lab, band) for k, v in preds.items()}   # reverse = label-free choice on every R arm
    avg = np.mean([v[1] for v in preds.values()], 0)
    aucs['phase_avg5'] = M.auc(avg, lab, band)
    vals = [aucs[k] for k in preds]
    out[w] = dict(aucs=aucs, sd=float(np.std(vals, ddof=1)), range=float(np.ptp(vals)), mean=float(np.mean(vals)),
                  elong_single=M.elongation(np.where(band, preds['R0_0'][1], 0), lab, band, 8.64),
                  elong_avg=M.elongation(np.where(band, avg, 0), lab, band, 8.64))
    for k, v in list(preds.items()) + [('phase_avg5', (None, avg))]:
        pooled.setdefault(k, []).append(M.block_hists(M.to_u8(v[1]), lab, band))
    print(w, json.dumps({k: round(v, 4) for k, v in aucs.items()}), 'sd %.4f range %.4f' % (out[w]['sd'], out[w]['range']))
    print(w, 'elongation single', out[w]['elong_single'], 'avg', out[w]['elong_avg'])
pa = {k: M.auc_from(np.concatenate([h[0] for h in v]).sum(0), np.concatenate([h[1] for h in v]).sum(0)) for k, v in pooled.items()}
single = [pa[k] for k in pa if k != 'phase_avg5']
out['pooled'] = dict(aucs=pa, sd=float(np.std(single, ddof=1)), range=float(np.ptp(single)))
print('pooled', json.dumps({k: round(v, 4) for k, v in pa.items()}), 'sd %.4f range %.4f' % (out['pooled']['sd'], out['pooled']['range']))
json.dump(out, open(f'{_P.OUT}/g1g2/phase_floor.json', 'w'), indent=1, default=float)
