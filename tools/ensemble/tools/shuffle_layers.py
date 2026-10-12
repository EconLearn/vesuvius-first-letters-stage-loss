#!/usr/bin/env python
"""Sabotage arm: copy a render zarr with its layers in a fixed random order (depth-shuffled).
usage: shuffle_layers.py IN.zarr OUT.zarr [--seed 0]"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ens import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
arr, attrs = C.open_render(a.src)
perm = np.random.default_rng(a.seed).permutation(arr.shape[0])
C.write_render(a.dst, np.asarray(arr[:])[perm], dict(attrs, sabotage=dict(depth_shuffled=True, seed=a.seed,
                                                                         perm=perm.tolist())))
print('wrote', a.dst, 'perm', perm.tolist())
