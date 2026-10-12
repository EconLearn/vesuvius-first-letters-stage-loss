#!/usr/bin/env python
"""stage2_q343: carry his 343 letter boxes (and the pre-registered null boxes) onto OUR q343 surface through 3D xyz,
then (with --run) render + sweep + ensemble + score there with exactly the stage-1 settings.

Mapping: pixels of every box on HIS joined mesh (lattice step 16 px) -> 3D points (our validated renderer's surface
sampling, grid scale from his meta) -> nearest valid vertex of OUR flattened tifxyz (KD-tree) -> refined to the
nearest render pixel of our surface in a +-20 px window -> per-box bounding box on our render grid.
Coverage gate (pre-registered): a box counts as covered when >= 50 % of its points have an our-surface point within
14 voxels (half the measured sheet pitch, G1G2 RCOV); stage 2 runs only if >= 50 % of the letter boxes are covered,
otherwise it exits 10 and reports the z ranges (our q343_trk band is z ~8525-9467; his letters are z 9515-10998).
Our surface's normal sign: render66 --normal-sign auto (umbilicus), as for any automatic surface.
usage: stage2_q343.py --rel REL --regions W/regions.json --out W/stage2 [--run] [--device cuda]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE))
sys.path.insert(0, str(LANE.parent))        # tools/: vcpaths.py, g12lib.py
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
from ens import common as C  # noqa: E402

OURS = Path(_P.WORK) / 'q343_trk' / 'flat' / 'q343_trk.tifxyz'    # our flattened q343 surface ($VC_WORK/q343_trk/flat)
NAME = 'concat_w047-w048_R5B2_z9500-11000'
RCOV = 14.0


def sh(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode:
        raise SystemExit(f'failed rc={r.returncode}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rel', required=True); ap.add_argument('--regions', required=True)
    ap.add_argument('--ours', default=str(OURS)); ap.add_argument('--out', required=True)
    ap.add_argument('--run', action='store_true'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--depths', default='-6..6:2')
    ap.add_argument('--sab-depths', default='-2..2:2')
    ap.add_argument('--margin', type=int, default=256, help='context px around the mapped boxes')
    ap.add_argument('--models', default='ensemble4')
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reg = json.loads(Path(a.regions).read_text())
    L = C.use_volume('PHerc0343')
    from scipy.spatial import cKDTree
    Ph, vh, mh = L.load_tifxyz(C.rel_file(a.rel, f'{NAME}.tifxyz')); Nh = L.vertex_normals(Ph); gh = float(mh['scale'][0])
    Po, vo, mo = L.load_tifxyz(a.ours); No = L.vertex_normals(Po); go = float(mo['scale'][0])
    vi = np.argwhere(vo)
    T = cKDTree(Po[vo])
    zr_ours = [float(np.nanmin(Po[..., 2])), float(np.nanmax(Po[..., 2]))]

    def map_boxes(boxes):
        """Per box: lattice points -> nearest our-vertex (KD, <= 60 vox) -> nearest our render pixel in a +-20 px
        window (refined 3D distance). Our surface can pass the same place twice (duplicate sheets), so the mapped
        pixels are reduced to the cluster around their median (|offset| <= 1.5 x box size per axis); the box is the
        inliers' extent. Covered = refined distance <= RCOV for >= 50 % of the points AND inliers >= 50 %."""
        res = []
        for r0, r1, c0, c1 in boxes:
            rows = np.arange(r0, r1, 16); cols = np.arange(c0, c1, 16)
            p, _ = L.surface_at_pixels(Ph, Nh, rows, cols, gh)
            p = p.reshape(-1, 3); p = p[np.isfinite(p).all(1)]
            if not len(p):
                res.append(dict(covered=False, frac=0.0)); continue
            d, j = T.query(p, distance_upper_bound=60.0)
            rec = dict(n=int(len(p)), z=[float(p[:, 2].min()), float(p[:, 2].max())],
                       vertex_dist_median=float(np.median(np.minimum(d, 60))))
            pix, dref = [], []
            for q, jj, dv in zip(p, j, d):
                if not np.isfinite(dv):
                    dref.append(np.inf); continue
                gi, gj = vi[jj]
                er, ec = int(round(gi / go - 0.5)), int(round(gj / go - 0.5))
                rr = np.arange(er - 20, er + 21, 2); cc = np.arange(ec - 20, ec + 21, 2)
                s, _ = L.surface_at_pixels(Po, No, rr, cc, go)
                dd = np.linalg.norm(s - q, axis=-1)
                if np.isfinite(dd).any():
                    k = np.nanargmin(dd); pix.append((rr[k // len(cc)], cc[k % len(cc)])); dref.append(float(dd.flat[k]))
                else:
                    dref.append(np.inf)
            dref = np.array(dref)
            ok = dref <= RCOV
            rec.update(frac=float(ok.mean()), dist_median=float(np.median(np.minimum(dref, 60))))
            if len(pix):
                pix = np.array(pix)
                med = np.median(pix, 0)
                inl = (np.abs(pix[:, 0] - med[0]) <= 1.5 * (r1 - r0)) & (np.abs(pix[:, 1] - med[1]) <= 1.5 * (c1 - c0))
                rec['inlier_frac'] = float(inl.mean())
                pi = pix[inl]
                rec['box_ours'] = [int(pi[:, 0].min()), int(pi[:, 0].max()) + 16,
                                   int(pi[:, 1].min()), int(pi[:, 1].max()) + 16]
            rec['covered'] = bool(ok.mean() >= 0.5 and rec.get('inlier_frac', 0) >= 0.5)
            res.append(rec)
        return res
    lm = map_boxes(reg['letters_raw'])
    nm = [map_boxes(b) for b in reg['nulls_raw']]
    zl = [x['z'] for x in lm if 'z' in x]
    rep = dict(ours=str(a.ours), our_z_range=zr_ours,
               his_letter_z_range=[min(z[0] for z in zl), max(z[1] for z in zl)] if zl else None,
               letters=lm, nulls=nm, coverage_rule=f'box covered if >= 50 % of points have an our-surface pixel within '
                                                    f'{RCOV} vox and >= 50 % map into one cluster; run if >= 50 % of '
                                                    f'letter boxes covered')
    ncov = sum(x['covered'] for x in lm)
    rep['letters_covered'] = f'{ncov}/{len(lm)}'
    (out / 'mapping.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(our_z_range=zr_ours, his_letter_z_range=rep['his_letter_z_range'],
                          letters_covered=rep['letters_covered'])), flush=True)
    if not lm or ncov < 0.5 * len(lm):
        print('NO COVERAGE: our q343 surface does not reach his letters; stage 2 needs a fit of OUR surface on his '
              'band (z 9500-11000) first.')
        return 10
    boxes = [x['box_ours'] for x in lm if x.get('box_ours')]
    if not boxes:
        print('no mapped letter boxes'); return 10
    nboxes = [[x['box_ours'] for x in s if x.get('box_ours')] for s in nm]
    allb = np.array(boxes + [b for s in nboxes for b in s])
    mg = a.margin
    r0 = max(0, (allb[:, 0].min() - mg) // 10 * 10); c0 = max(0, (allb[:, 2].min() - mg) // 10 * 10)
    r1 = -(-(allb[:, 1].max() + mg) // 10) * 10; c1 = -(-(allb[:, 3].max() + mg) // 10) * 10
    if (r1 - r0) * (c1 - c0) > 40e6:
        print(f'canvas {r1 - r0} x {c1 - c0} px is implausibly large (> 40 Mpx): mapping scattered; not rendering')
        return 11
    reg2 = dict(letters_raw=boxes, nulls_raw=nboxes, canvas=dict(rows=[int(r0), int(r1)], cols=[int(c0), int(c1)]),
                source='mapped from his boxes through 3D xyz (stage2_q343.py)')
    (out / 'regions_ours.json').write_text(json.dumps(reg2, indent=1))
    if not a.run:
        print('mapping written; re-run with --run to render + sweep + score'); return 0
    py = C.PY; R = out / 'q343_letters.zarr'
    sh([py, LANE / 'render66.py', '--scroll', 'PHerc0343', '--tifxyz', a.ours, '--rows', f'{r0}:{r1}', '--cols',
        f'{c0}:{c1}', '--normal-sign', 'auto', '--out', R, '--save-pts', '--procs', '8'])
    sh([py, LANE / 'tools' / 'shuffle_layers.py', R, out / 'q343_letters_shuffled.zarr'])
    run = (['bash', str(LANE / 'sweep_locked.sh'), 'stage2'] if a.device.startswith('cuda') else [py, LANE / 'depth_sweep.py'])
    sh(run + ['--render', R, '--models', a.models, '--depths', a.depths, '--order', 'reverse', '--phase', '0,0',
              '--phase', '40,200', '--merge', 'mean', '--out', out / 'sweep', '--device', a.device])
    sh(run + ['--render', out / 'q343_letters_shuffled.zarr', '--models', a.models, '--depths', a.sab_depths,
              '--order', 'reverse', '--out', out / 'sweep_shuffled', '--device', a.device])
    sh([py, LANE / 'ensemble_maps.py', '--render', R, '--maps', out / 'sweep', '--models', a.models, '--tag', 'merged', '--stat-depths',
        a.depths, '--write-depths', a.depths, '--combine', 'mean', '--out', out / 'ens', '--name', 'q343'])
    sh([py, LANE / 'ensemble_maps.py', '--render', out / 'q343_letters_shuffled.zarr', '--maps', out / 'sweep_shuffled',
        '--models', a.models, '--stat-depths', a.sab_depths, '--write-depths', a.sab_depths, '--combine', 'mean', '--out', out / 'ens_shuffled',
        '--name', 'q343', '--no-png'])
    sh([py, LANE / 'score_regions.py', '--ens', out / 'ens', '--name', 'q343', '--regions', out / 'regions_ours.json',
        '--offset', f'{r0},{c0}', '--sabotage-ens', out / 'ens_shuffled', '--out', out / 'score.json'])
    sh([py, LANE / 'display.py', '--render', R, '--ens-dir', out / 'ens', '--name', 'q343', '--ink', 'dmean',
        '--ct-d', '1', '--out', out / 'q343_letters_display.png', '--strips', out / 'strips', '--strip-h', '450'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
