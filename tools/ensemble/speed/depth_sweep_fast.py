#!/usr/bin/env python
"""depth_sweep.py (unchanged) with an OPT-IN GPU fast path for the villa family.

  depth_sweep_fast.py --villa-fast  <any depth_sweep.py arguments>

* Without --villa-fast this runs exactly depth_sweep.main() (the validated default path).
* With --villa-fast, ink9_s42 / ink9_s43 / dnative use speed/fastvilla.FastVillaAdapter: the tile stack goes to the
  GPU once; the zero-patch filter, patch extraction, robust-MAD normalisation (exact re-implementation, numpy's
  MAD-fallback std still on the CPU) and the Hann overlap-add run on the GPU. Hecate and v8-in are unchanged.
  Each map's .json gets "villa_path": "speed/fastvilla.FastVillaAdapter".
* Exactness (2026-10-11, RTX 4060, torch 2.12.1+cu130, numpy 2.5.3): maps bit-identical to the default path on
  30.0 M px (dnative + ink9_s42, d -2/0/+2, 16 and 6 CPUs), on the 343 letter region (phase 40,200, tile 1024) and on
  a dense q343_trk band canvas. Re-run tests/test_norm_exact.py --device cuda after any numpy,
  torch or villa update: the normalisation mirrors numpy's float semantics.
* GPU runs still refuse unless started under gpu_locked.sh (ENS_GPU_LOCK_HELD=1) with no fit_spiral.py running
  (depth_sweep.gpu_guard, unchanged)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # tools/ensemble (depth_sweep.py, ens/)
sys.path.insert(0, str(HERE))
import depth_sweep as DS  # noqa: E402
from ens import adapters as A  # noqa: E402

if __name__ == '__main__':
    if '--villa-fast' in sys.argv:
        sys.argv.remove('--villa-fast')
        import fastvilla as FV
        A.FAMILIES['villa'] = FV.FastVillaAdapter
        _run_one = DS.run_one

        def run_one(ad, *a, **k):
            mp, st = _run_one(ad, *a, **k)
            if st == 'done' and getattr(ad, 'fast', False):
                j = mp.with_suffix('.json'); meta = json.loads(j.read_text())
                meta['villa_path'] = 'speed/fastvilla.FastVillaAdapter'
                j.write_text(json.dumps(meta, indent=1))
            return mp, st
        DS.run_one = run_one
        print('depth_sweep_fast: villa GPU fast path ON', file=sys.stderr, flush=True)
    sys.exit(DS.main())
