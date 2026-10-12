#!/usr/bin/env python
"""tools/ensemble/depth_sweep.py unchanged, for the cloud job scripts.

depth_sweep --device cuda refuses unless it was started through sweep_locked.sh (shared GPU: GPU lock + no
fit_spiral.py). On a rented single-GPU host the job scripts serialise GPU work themselves, so this wrapper sets
ENS_GPU_LOCK_HELD=1. The 'no fit_spiral.py running' check is kept unless POD_ALLOW_FIT_OVERLAP=1 (the optional pipelined
mode, where the next scroll's fit deliberately shares the 24 GB card with this sweep).
usage: sweep.py [depth_sweep args...]"""
import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2]                       # tools/
ENS = Path(os.environ.get('ENS_LANE', str(TOOLS / 'ensemble')))   # tools/ensemble
sys.path.insert(0, str(ENS))
os.environ['ENS_GPU_LOCK_HELD'] = '1'
import depth_sweep as DS  # noqa: E402

if os.environ.get('POD_ALLOW_FIT_OVERLAP') == '1':
    DS.gpu_guard = lambda device: None

if os.environ.get('VILLA_FAST') == '1':
    # Opt-in GPU fast path for the villa family (ensemble/speed/fastvilla.py; bit-identical to the default path,
    # re-check with speed/tests/compare_maps.py on a new torch/numpy). Each map's .json gets "villa_path".
    import json
    sys.path.insert(0, os.environ.get('SPEED_DIR', str(ENS / 'speed')))
    from ens import adapters as A  # noqa: E402
    import fastvilla as FV  # noqa: E402
    A.FAMILIES['villa'] = FV.FastVillaAdapter
    _run_one = DS.run_one

    def _run_one_tagged(ad, *a, **k):
        mp, st = _run_one(ad, *a, **k)
        if st == 'done' and getattr(ad, 'fast', False):
            j = mp.with_suffix('.json'); meta = json.loads(j.read_text())
            meta['villa_path'] = 'speed/fastvilla.FastVillaAdapter'
            j.write_text(json.dumps(meta, indent=1))
        return mp, st
    DS.run_one = _run_one_tagged

if __name__ == '__main__':
    sys.exit(DS.main(sys.argv[1:]))
