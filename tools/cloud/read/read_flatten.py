"""Official flatten_spiral_checkpoint.main() unchanged (= tools/g12_flatten.py), with the umbilicus as an argument so it
serves every scroll; keeps the exported unflattened source surface next to the output.
Refuses a checkpoint whose completed_iterations differ from EXPECT_IT (default 10000; 0 = no check).
VOXEL_UM = voxel size of the scroll in um (default $VOXEL_UM, else 8.64); jobs pass the screening config's voxel_um.
usage: read_flatten.py CKPT OUTPUT.tifxyz UMBILICUS.json [EXPECT_IT [VOXEL_UM]]"""
import os, shutil, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)
sys.path.insert(0, _P.SPIRAL_FIT_DIR)
import flatten_spiral_checkpoint as F

ckpt, out, umb = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
expect = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
voxel = sys.argv[5] if len(sys.argv) > 5 else os.environ.get('VOXEL_UM', '8.64')
it = int(torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)['completed_iterations'])
print(f'[read] checkpoint {ckpt} at iteration {it}', flush=True)
if expect and it != expect:
    raise SystemExit(f'ABORT: checkpoint at {it}, expected {expect}')
out.parent.mkdir(parents=True, exist_ok=True)
_orig = F._export_source_surface


def _keep(*a, **k):
    t = time.time()
    surf = _orig(*a, **k)
    dst = out.parent / (out.name.replace('.tifxyz', '') + '.source.tifxyz')
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(surf, dst)
    print(f'[read] kept source surface -> {dst} (export {time.time() - t:.0f}s)', flush=True)
    return surf


F._export_source_surface = _keep
t0 = time.time()
rc = F.main([ckpt, str(out), '--umbilicus', umb, '--voxel-size-um', str(voxel)])
print(f'[read] flatten total {time.time() - t0:.0f}s rc={rc}', flush=True)
sys.exit(rc or 0)
