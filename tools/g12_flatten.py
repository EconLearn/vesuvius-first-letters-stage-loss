"""Run the official flatten_spiral_checkpoint.main() unchanged, but keep a copy of the exported (unflattened)
source surface next to the output (<output>.source.tifxyz) for the auto_raw arm.
usage: g12_flatten.py CKPT OUTPUT.tifxyz  (env UMBILICUS overrides the umbilicus JSON; voxel size 8.64 um = PHerc1447)"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, _P.SPIRAL_FIT_DIR)
import flatten_spiral_checkpoint as F

ckpt, out = sys.argv[1], Path(sys.argv[2]).resolve()
_orig = F._export_source_surface


def _keep(*a, **k):
    t = time.time()
    surf = _orig(*a, **k)
    dst = out.parent / (out.name.replace('.tifxyz', '') + '.source.tifxyz')
    shutil.copytree(surf, dst)
    print(f'[g12] kept source surface -> {dst} (export {time.time() - t:.0f}s)', flush=True)
    return surf


F._export_source_surface = _keep
t0 = time.time()
rc = F.main([ckpt, str(out), '--umbilicus', os.environ.get('UMBILICUS', f'{_P.DATA}/spiral/PHerc1447/umbilicus.json'), '--voxel-size-um', '8.64'])
print(f'[g12] flatten total {time.time() - t0:.0f}s rc={rc}', flush=True)
