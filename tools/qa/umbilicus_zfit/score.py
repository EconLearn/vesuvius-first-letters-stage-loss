"""Score RAW / FIT / SABOTAGE umbilici against the references (PREREG_zfit.txt).
usage: score.py DATA dev            -> dev scrolls only, every H of the grid (chooses H)
       score.py DATA final H_MM     -> every scroll at the pre-registered H; ship rule; writes DATA/final_H.json"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from zfit import zfit

D = Path(sys.argv[1]); MODE = sys.argv[2]
S = json.loads((D / 'scrolls.json').read_text()); RAW = json.loads((D / 'raw.json').read_text())
DEV = ['PHerc0358', 'PHerc1203', 'PHerc0191']; TEST = [s for s in S if s not in DEV]
SEEDS = sorted(next(iter(RAW.values()))['0'], key=int)   # 1..9, or 11..19 for the replication
GRID_MM = [250 * 9.362e-3 * 2 ** k for k in range(5)]   # 2.34 .. 37.4 mm


def ref_at(s, z):
    u = json.loads(Path(S[s]['ref']).read_text())
    P = np.array(sorted([[c['z'], c['x'], c['y']] for c in u['control_points']]), float)
    return np.stack([np.interp(z, P[:, 0], P[:, 1]), np.interp(z, P[:, 0], P[:, 2])], 1)


def errors(s, beta, arm, H_mm, repeats):
    """Per-slice error in mm, for one scroll (repeats 1: mean over the 9 single-run outputs)."""
    f = S[s]['level_factor']; vox = S[s]['vox_um'] / 1000
    z = np.array(S[s]['z'], float) * f; ref = ref_at(s, z)
    est = np.array([RAW[s][beta][sd] for sd in SEEDS], float) * f        # (9, n, 2), reference frame
    H = None if H_mm is None else H_mm / vox
    zlab = z.copy()
    if arm == 'sab':
        zlab = z[np.random.default_rng(7).permutation(len(z))]

    def curve(runs):                                                       # runs: (r, n, 2)
        if arm == 'raw':
            return np.median(runs, 0)
        pts = np.concatenate([np.column_stack([zlab, r]) for r in runs])
        return zfit(pts, z, H)
    if repeats == 1:
        e = np.mean([np.linalg.norm(curve(est[i:i + 1]) - ref, axis=1) for i in range(len(SEEDS))], 0)
    else:
        e = np.linalg.norm(curve(est) - ref, axis=1)
    return e * vox


if MODE == 'dev':
    for beta in ('1.5', '0'):
        base = np.concatenate([errors(s, beta, 'raw', None, 1) for s in DEV])
        print(f'uncrush {beta} repeats 1 RAW dev median {np.median(base):.3f} mm')
        for H in GRID_MM:
            e = np.concatenate([errors(s, beta, 'fit', H, 1) for s in DEV])
            print(f'  H {H:6.2f} mm: FIT dev median {np.median(e):.3f}  per scroll ' +
                  ' '.join(f'{s[5:]}:{np.median(errors(s, beta, "fit", H, 1)):.2f}' for s in DEV))
    sys.exit()

H = float(sys.argv[3]); res = {}
for beta in ('1.5', '0'):
    for rep in (1, 9):
        tab = {s: {a: errors(s, beta, a, None if a == 'raw' else H, rep) for a in ('raw', 'fit', 'sab')} for s in S}
        res[f'{beta}_r{rep}'] = {s: {a: v.tolist() for a, v in t.items()} for s, t in tab.items()}
        med = {a: np.array([np.median(tab[s][a]) for s in TEST]) for a in ('raw', 'fit', 'sab')}
        p = wilcoxon(med['fit'], med['raw'], alternative='less').pvalue
        worst = float(np.max(med['fit'] - med['raw']))
        beats_sab = int(np.sum(med['fit'] < med['sab']))
        sl = {a: np.concatenate([tab[s][a] for s in TEST]) for a in ('raw', 'fit', 'sab')}
        ok = np.median(med['fit']) < np.median(med['raw']) and p < 0.05 and worst <= 0.25 and beats_sab >= 9
        print(f'\n== uncrush {beta}, repeats {rep}, H {H:.2f} mm, held-out 13 scrolls ==')
        print(f'median of scroll medians: RAW {np.median(med["raw"]):.3f}  FIT {np.median(med["fit"]):.3f}  SAB {np.median(med["sab"]):.3f} mm'
              f' | scroll Wilcoxon p {p:.4g} | worst scroll {worst:+.3f} mm | FIT beats SAB on {beats_sab}/13'
              + (f' | RULE {"PASS" if ok else "FAIL"}' if rep == 1 else ''))
        print(f'slices n={len(sl["raw"])}: median RAW {np.median(sl["raw"]):.3f} FIT {np.median(sl["fit"]):.3f} SAB {np.median(sl["sab"]):.3f};'
              f' p90 RAW {np.percentile(sl["raw"], 90):.3f} FIT {np.percentile(sl["fit"], 90):.3f};'
              f' better {np.sum(sl["fit"] < sl["raw"] - 0.05)} worse {np.sum(sl["fit"] > sl["raw"] + 0.05)}')
        for s in S:
            print(f'  {s:12s} {"dev " if s in DEV else "test"} {S[s]["ref_kind"]:10s} n={len(tab[s]["raw"]):3d}  RAW {np.median(tab[s]["raw"]):.3f}'
                  f'  FIT {np.median(tab[s]["fit"]):.3f}  SAB {np.median(tab[s]["sab"]):.3f}')
(D / f'final_{H:.2f}.json').write_text(json.dumps(res))
