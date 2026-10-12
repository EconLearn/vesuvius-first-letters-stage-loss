"""Print the tags of running quiet_super.py supervisors whose job has a PARTIAL checkpoint (a resume is pending or a fit
of a started job may relaunch). Empty output = no supervisor can (re)launch a started fit."""
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)
RUNS = os.environ.get('QUIET_RUNS', os.path.join(os.environ.get('QUIET_WORK', os.path.join(_P.WORK, 'quiet')), 'runs'))
out = subprocess.run(['pgrep', '-af', 'quiet_super.py'], capture_output=True, text=True).stdout.splitlines()
pend = []
for line in out:
    parts = line.split()
    if 'quiet_super.py' not in line or 'pgrep' in line or 'pending_resume' in line:
        continue
    try:
        i = [k for k, p in enumerate(parts) if p.endswith('quiet_super.py')][0]
        tag, steps = parts[i + 1], int(parts[i + 3])
    except Exception:
        continue
    ck = f'{RUNS}/{tag}/checkpoint_fitted.ckpt'
    if not os.path.exists(ck):
        continue
    import torch
    try:
        it = int(torch.load(ck, map_location='cpu', weights_only=False, mmap=True)['completed_iterations'])
    except Exception:
        it = -1
    if it < steps:
        pend.append(f'{tag}:{it}/{steps}')
print(' '.join(pend))
