"""Fit supervisor: the official fit_spiral.py run with the G1 recipe's base overrides (tools/g12_fit_run.sh), VRAM guards
and autosave-resume logic, for unlabelled scrolls:
  - no organiser-surface scoring (there are no labels); autosaves are only logged
  - extra launch guards: no other fit_spiral.py process may be running (one GPU fit at a time), and the disk must keep
    >= $VC_MIN_FREE_GB (default 20) above this fit's need (vlib.guards.free_gb)
  - gives up on a fit after MAX_KILLS VRAM-cap stops (no infinite stop/resume loop), logging why
  - keeps the final checkpoint (no --delete-final)
Runs: $QUIET_RUNS/<tag>/ (default $QUIET_WORK/runs) ; logs: $QUIET_WORK/logs/fit_<tag>.{log,super}
(QUIET_WORK default $VC_WORK/quiet; the fit itself runs in $SPIRAL_FIT_DIR with $POD_PYX, default $VC_ROOT/.venv-x2/bin/python)

usage: quiet_super.py TAG DATASET STEPS RESUME|none EXTRA_JSON [--start-max 4300] [--kill 7650] [--idle-util 50]
                      [--idle-s 60] [--max-kills 3] [--scroll-spec PATH]
--scroll-spec is passed to fit_spiral.py (default: <dataset>/spiral-scroll.json); job 3 uses it to fit one band under
each spiral-scroll.<SENSE>.json variant when the winding sense is unknown (see pick_sense.py).
"""
import sys, os, time, json, signal, subprocess, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # tools/: vcpaths.py, vlib/
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)
from vlib.guards import free_gb  # noqa: E402

ap = argparse.ArgumentParser()
for k in ('tag', 'dataset', 'steps', 'resume', 'extra'):
    ap.add_argument(k)
ap.add_argument('--start-max', type=float, default=4300)
ap.add_argument('--kill', type=float, default=7650)
ap.add_argument('--idle-util', type=float, default=50)
ap.add_argument('--idle-s', type=float, default=60)
ap.add_argument('--max-kills', type=int, default=3)
ap.add_argument('--disk-need-gb', type=float, default=4.0)
ap.add_argument('--sf', default=_P.SPIRAL_FIT_DIR)
ap.add_argument('--scroll-spec', default=None)
a = ap.parse_args()
STEPS = int(a.steps)
QW = Path(os.environ.get('QUIET_WORK', os.path.join(_P.WORK, 'quiet')))
RUN = Path(os.environ.get('QUIET_RUNS', str(QW / 'runs'))) / a.tag; RUN.mkdir(parents=True, exist_ok=True)
CACHE = QW / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
L = QW / 'logs'; LOG = L / f'fit_{a.tag}.log'
CK = RUN / 'checkpoint_fitted.ckpt'
# base overrides of the G1 recipe (tools/g12_fit_run.sh); EXTRA carries z band, F10 and the tracks keys
BASE = {"z_begin": 13100, "z_end": 14100,
        "input_use_verified_patches": False, "input_use_tracks": False, "input_use_fibers": False,
        "input_use_fiber_directions": False, "input_use_pcl_absolute": False, "input_use_pcl_relative": False,
        "input_use_pcl_same_winding": False, "input_use_pcl_drawn_control_points": False,
        "input_use_winding_inference": False, "input_use_outer_shell": False,
        "input_use_normals": True, "input_use_gradient_magnitude": True,
        "dense_spacing_mode": "grad_mag", "loss_weight_dense_spacing": 0, "loss_weight_shell_outer": 0}
OV = dict(BASE); OV.update(json.loads(a.extra)); OV['optimizer_num_training_steps'] = STEPS


def say(*x):
    print(time.strftime('%Y-%m-%dT%H:%M:%S'), *x, flush=True)


def smi(q):
    r = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return float(r.stdout.strip().split('\n')[0])


def other_fits():
    r = subprocess.run(['pgrep', '-f', 'fit_spiral.py'], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.strip()]


def ck_iters(p):
    import torch
    try:
        return int(torch.load(str(p), map_location='cpu', weights_only=False, mmap=True)['completed_iterations'])
    except Exception:
        return None


GPU_LOCK = os.environ.get('GPU_LOCK', str(QW / 'GPU_LOCK'))  # other GPU work (gpu/gpu_locked.sh) holds this; no (re)launch meanwhile


def gpu_locked():
    try:
        pid = int(open(GPU_LOCK).read().split()[0])
    except FileNotFoundError:
        return False
    except Exception:
        return True                      # unreadable lock: honour it
    return os.path.exists(f'/proc/{pid}')   # a lock whose owner died is stale and ignored


def wait_for_gpu(honour_lock=True):
    ok_since = None
    lock_said = False
    while True:
        u, ut, others = smi('memory.used'), smi('utilization.gpu'), other_fits()
        disk_ok = free_gb() - a.disk_need_gb >= _P.MIN_FREE_GB
        locked = honour_lock and gpu_locked()   # a resume of a started job ignores the lock (no deadlock)
        if locked != lock_said:
            say('GPU_LOCK held by other GPU work; waiting' if locked else 'GPU_LOCK released'); lock_said = locked
        if u <= a.start_max and ut <= a.idle_util and not others and disk_ok and not locked:
            ok_since = ok_since or time.time()
            if time.time() - ok_since >= a.idle_s:
                say('gpu idle, no other fit, disk ok; starting', u, ut, round(free_gb(), 1)); return
        else:
            if not disk_ok:
                say('DISK GUARD:', _P.disk_guard_path(), 'free', round(free_gb(), 1), 'GB; need', a.disk_need_gb, 'above the', _P.MIN_FREE_GB, 'GB floor; waiting')
                time.sleep(290)
            ok_since = None
        time.sleep(10)


def launch(resume):
    env = dict(os.environ)
    env.update({'FIT_SPIRAL_TRITON': os.environ.get('FIT_SPIRAL_TRITON', '0'), 'WANDB_MODE': 'disabled', 'FIT_SPIRAL_RUN_DIR': str(RUN),
                'FIT_SPIRAL_CACHE_DIR': str(CACHE), 'FIT_SPIRAL_CONFIG_OVERRIDES': json.dumps(OV)})
    env.pop('FIT_SPIRAL_RESUME_PATH', None)
    if resume and resume != 'none':
        env['FIT_SPIRAL_RESUME_PATH'] = str(resume)
    f = open(LOG, 'a')
    f.write(f'\n==== START {time.strftime("%Y-%m-%dT%H:%M:%S")} resume={resume} overrides={json.dumps(OV)}\n'); f.flush()
    p = subprocess.Popen([os.environ.get('POD_PYX', os.path.join(_P.ROOT, '.venv-x2', 'bin', 'python')), '-u', 'fit_spiral.py', '--dataset', a.dataset]
                         + (['--scroll-spec', a.scroll_spec] if a.scroll_spec else []),
                         cwd=a.sf, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    (L / f'fit_{a.tag}.pid').write_text(str(p.pid))
    say('launched', a.tag, 'pid', p.pid, 'resume', resume)
    return p


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait()


resume, kills, last_m = a.resume, 0, None
while True:
    if CK.exists():
        it = ck_iters(CK)
        if it is not None and it >= STEPS:
            say('final checkpoint present, done'); break
        if it is not None and it > 0:
            resume = str(CK)
    wait_for_gpu(honour_lock=(resume in (None, 'none')))
    p = launch(resume)
    hot = 0
    while p.poll() is None:
        time.sleep(15)
        u = smi('memory.used')
        hot = hot + 1 if u > a.kill else 0
        if hot >= 2:
            kills += 1
            say('GPU over cap', u, f'-> stopping my fit ({kills}/{a.max_kills}); will resume from autosave')
            stop(p); break
        if CK.exists() and CK.stat().st_mtime != last_m:
            time.sleep(15); last_m = CK.stat().st_mtime
            it_now = ck_iters(CK)
            say('autosave', it_now)
            if it_now is not None and it_now >= STEPS and p.poll() is None:
                say('final checkpoint written; stopping post-fit metrics to free the GPU'); stop(p); break
    rc = p.poll(); say('fit exited rc', rc)
    time.sleep(5)
    if CK.exists() and (ck_iters(CK) or 0) >= STEPS:
        say('done; final checkpoint kept at', CK); break
    if kills >= a.max_kills:
        say(f'GAVE UP after {kills} VRAM-cap stops; autosave kept at', CK); sys.exit(3)
    if rc not in (None, 0, -15, -9) and hot < 2:
        tail = LOG.read_text()[-1500:]
        if 'satisfaction' in tail or 'vc_spiral' in tail:
            say('exited in post-fit metrics; checkpoint kept'); break
        say('fit crashed; see', LOG); sys.exit(2)
say('supervisor exit')
