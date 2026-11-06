"""Before/after figure for one scroll: the x and y of the umbilicus along z, reference vs today's single-run output vs
--zfit, plus the per-slice error. usage: figure.py DATA SCROLL BETA H_MM SEED OUT.png"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zfit import zfit

D = Path(sys.argv[1]); s = sys.argv[2]; beta = sys.argv[3]; H_mm = float(sys.argv[4]); seed = sys.argv[5]; OUT = sys.argv[6]
S = json.loads((D / 'scrolls.json').read_text())[s]; RAW = json.loads((D / 'raw.json').read_text())[s][beta]
f = S['level_factor']; vox = S['vox_um'] / 1000
z = np.array(S['z'], float) * f; est = np.array(RAW[seed], float) * f
u = json.loads(Path(S['ref']).read_text())
P = np.array(sorted([[c['z'], c['x'], c['y']] for c in u['control_points']]), float)
ref = np.stack([np.interp(z, P[:, 0], P[:, 1]), np.interp(z, P[:, 0], P[:, 2])], 1)
fit = zfit(np.column_stack([z, est]), z, H_mm / vox)
zmm = z * vox
fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for k, name in enumerate('xy'):
    ax[k].plot(P[:, 0] * vox, P[:, k + 1] * vox, color='0.3', lw=2, label='hand-made reference')
    ax[k].plot(zmm, est[:, k] * vox, 'o', ms=3, color='tab:red', label='today (one run, --repeats 1)')
    ax[k].plot(zmm, fit[:, k] * vox, '-', color='tab:blue', lw=1.5, label=f'--zfit ({H_mm:.0f} mm window)')
    ax[k].set_ylabel(f'{name} (mm)')
    lo, hi = np.percentile(ref[:, k] * vox, [0, 100]); ax[k].set_ylim(lo - 6, hi + 6)
ax[0].legend(fontsize=8, loc='best')
ax[2].plot(zmm, np.linalg.norm(est - ref, axis=1) * vox, 'o', ms=3, color='tab:red', label='today')
ax[2].plot(zmm, np.linalg.norm(fit - ref, axis=1) * vox, '-', color='tab:blue', label='--zfit')
ax[2].set_ylabel('distance to reference (mm)'); ax[2].set_xlabel('z (mm)'); ax[2].set_ylim(0, None)
ax[0].set_title(f'{s}: umbilicus from the published m7 normal grids, uncrush {beta}, seed {seed}')
fig.tight_layout(); fig.savefig(OUT, dpi=110)
print('wrote', OUT)
