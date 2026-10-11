"""Learning-curve table for the automatic arms only (fixed depth order 'forward', e2e coverage cap 14 vox), with
pooled AUC and paired block-bootstrap CIs vs A (organiser render) and R (organiser surface, our render).
usage: g12_auto_table.py TAG [TAG...]"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, _P.TOOLS)
import g12_g2 as G
from vlib import metrics as M
ref = {}
geo = {}
for w in G.WINDS:
    g = np.load(G.W / 'org' / w / 'geo.npz'); band = g['band']; lab = g['label'].astype(bool) & band
    r0, r1, c0, c1 = g['sub']; geo[w] = (g, band, lab)
    a = G._pred(G.E1 / f'1447_{w}' / 'pred' / 's42_A_reverse.tif')[r0:r1, c0:c1]
    rr = G.load_pair(f'{w}_R')[1]
    ref[w] = {'A': M.block_hists(M.to_u8(a), lab, band), 'R': M.block_hists(M.to_u8(rr), lab, band)}
out = {}
for tag in sys.argv[1:]:
    pos, pf, pr, _, _ = G.surface_arm(f'auto_{tag}', 'w058')
    T = cKDTree(pos, balanced_tree=False, compact_nodes=False)
    rec = {}; hs = {'fwd': [], 'sep': []}
    for w in G.WINDS:
        g, band, lab = geo[w]
        Q = g['pts'][band]
        d, j = T.query(Q, distance_upper_bound=G.RMAX, workers=4); hit = np.isfinite(d); jf = np.where(hit, j, 0)
        cov = hit & (d <= G.RCOV)
        CF = np.zeros(band.shape, np.float32); CR = np.zeros(band.shape, np.float32)
        CF[band] = np.where(cov, pf[jf], 0); CR[band] = np.where(cov, pr[jf], 0)
        r, p = G.score_canvas(CF, CR, lab, band, c2=True, order='forward')
        rec[w] = dict(auc=r['auc'], auc_sep_rule=r['auc_sep_rule'], chosen_sep=r['chosen_sep'], auc_oracle=r['auc_oracle'],
                      ci95=r['ci95'], c2=r.get('c2_median'), elong=r['elong'], le2=float((hit & (d <= 2)).mean()),
                      le6=float((hit & (d <= 6)).mean()), d_median=float(np.median(d[hit])))
        hs['fwd'].append(M.block_hists(M.to_u8(p), lab, band))
        psep = CF if r['chosen_sep'] == 'forward' else CR
        hs['sep'].append(M.block_hists(M.to_u8(psep), lab, band))
    def pool(lst):
        return (np.concatenate([x[0] for x in lst]), np.concatenate([x[1] for x in lst]))
    hf = pool(hs['fwd']); hsp = pool(hs['sep'])
    rec['pooled'] = dict(auc=M.auc_from(hf[0].sum(0), hf[1].sum(0)), auc_sep_rule=M.auc_from(hsp[0].sum(0), hsp[1].sum(0)))
    for k in ('A', 'R'):
        hk = pool([ref[w][k] for w in G.WINDS])
        rec['pooled'][f'd_vs_{k}'] = rec['pooled']['auc'] - M.auc_from(hk[0].sum(0), hk[1].sum(0))
        rec['pooled'][f'd_vs_{k}_ci95'] = G.paired_ci(hk, hf)
    out[tag] = rec
    print(tag, 'w058 %.4f  w060 %.4f  pooled %.4f  (sep-rule %.4f)  dR %+.4f %s  le6 %.3f/%.3f  le2 %.3f/%.3f  c2 %.3f/%.3f' % (
        rec['w058']['auc'], rec['w060']['auc'], rec['pooled']['auc'], rec['pooled']['auc_sep_rule'], rec['pooled']['d_vs_R'],
        np.round(rec['pooled']['d_vs_R_ci95'], 4).tolist(), rec['w058']['le6'], rec['w060']['le6'], rec['w058']['le2'], rec['w060']['le2'],
        rec['w058']['c2'], rec['w060']['c2']), flush=True)
json.dump(out, open(f'{_P.OUT}/g1g2/auto_table.json', 'w'), indent=1, default=float)
