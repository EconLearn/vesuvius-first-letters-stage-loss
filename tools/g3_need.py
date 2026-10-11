#!/usr/bin/env python
"""NEED masks for g3_v8.py: the canvas pixels the G3 scorer reads (label-free: uses the scoring support only,
never ink labels or predictions).
  auto arm (ARMDIR/auto_<tag>.json + <canvas>_pts.npy): nearest canvas pixel (3D) of every organiser band pixel
      (certainty mask ∩ z band, w058 + w060) with distance <= CAP (14 vox = the e2e coverage cap) -> needed.
  organiser arm <w>_R: the band mask itself (direct scoring).
usage: g3_need.py auto TAG [ARMDIR] | g3_need.py org      (env G3BAND=b2: band-2 bench, w060 only, names prefixed b2)
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

import os
G1 = Path(f'{_P.WORK}/g1g2'); NEED = Path(f'{_P.WORK}/g3/need')
WINDS = ['w058', 'w060']; CAP = 14.0; PFX = ''
if os.environ.get('G3BAND') == 'b2':
    G1 = Path(f'{_P.WORK}/g3/b2'); WINDS = ['w060']; PFX = 'b2'


def auto(tag, armdir):
    armdir = Path(armdir)
    meta = json.loads((armdir / f'auto_{tag}.json').read_text())
    P, IDX, shapes = [], [], []
    for k, c in enumerate(meta['comps']):
        pos = np.load(armdir / f'{c["name"]}_pts.npy'); shapes.append(pos.shape[:2])
        ok = np.isfinite(pos).all(-1)
        rc = np.argwhere(ok)
        P.append(pos[ok]); IDX.append(np.column_stack([np.full(len(rc), k), rc]))
    P = np.concatenate(P); IDX = np.concatenate(IDX)
    T = cKDTree(P, balanced_tree=False, compact_nodes=False)
    need = [np.zeros(s, bool) for s in shapes]
    for w in WINDS:
        g = np.load(G1 / 'org' / w / 'geo.npz')
        Q = g['pts'][g['band']]
        d, j = T.query(Q, distance_upper_bound=CAP, workers=8)
        j = j[np.isfinite(d)]
        for k, r, c in IDX[np.unique(j)]:
            need[k][r, c] = True
        print(w, 'band px', len(Q), 'covered <=14', int(np.isfinite(d).sum()), flush=True)
    NEED.mkdir(parents=True, exist_ok=True)
    tot = 0
    for k, c in enumerate(meta['comps']):
        np.save(NEED / f'{c["name"]}.npy', need[k]); tot += int(need[k].sum())
    print(tag, 'needed px', tot, 'of canvas px', sum(int(np.prod(s)) for s in shapes), flush=True)


def org():
    NEED.mkdir(parents=True, exist_ok=True)
    for w in WINDS:
        g = np.load(G1 / 'org' / w / 'geo.npz')
        np.save(NEED / f'{PFX}{w}_R.npy', g['band'].astype(bool))
        print(w, 'R need px', int(g['band'].sum()), 'of', g['band'].size)


if __name__ == '__main__':
    if sys.argv[1] == 'auto':
        auto(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else G1 / 'arms')
    else:
        org()
