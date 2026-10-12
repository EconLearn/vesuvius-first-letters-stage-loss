#!/usr/bin/env python
"""Pod cost guard. Pod-hours are wall-clock hours since the pod (container PID 1) started, because RunPod bills the
whole time the pod is up, busy or idle. Dollars = hours x (GPU $/h + storage $/h). The rates are guesses until the
operator sets the real $/h shown in the RunPod UI (`cost_guard.py init --rate X --force-rates`).

  init   [--budget 25] [--rate 0.34] [--storage-gb 250] [--t0 EPOCH]   writes STATE/budget.json once (t0 kept)
  check  LABEL [--est-h H]   print elapsed/spent; exit 3 if spent >= budget - reserve, exit 4 if spent + H*rate would
                             cross it (the step is not started); exit 0 otherwise
  status                     print one line
  watch  PGID                every 60 s; at budget - reserve: SIGTERM the process group, log BUDGET STOP, exit 3
The guard never stops the pod itself (no provider API key on the pod, and an unverified self-shutdown is not trusted):
on a BUDGET STOP, sync the results back and stop/terminate the pod yourself.
STATE = $POD_STATE (default $VC_ROOT/cloud_state, see tools/vcpaths.py).
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

STATE = Path(os.environ.get('POD_STATE', os.path.join(_P.ROOT, 'cloud_state')))
CFG = STATE / 'budget.json'
STATUS = STATE / 'STATUS.log'
STORAGE_USD_PER_GB_MONTH = 0.10      # RunPod running-pod disk price (guess; check the UI)


def pid1_start_epoch():
    try:
        btime = int([l for l in open('/proc/stat') if l.startswith('btime')][0].split()[1])
        st = open('/proc/1/stat').read().rsplit(')', 1)[1].split()
        return btime + int(st[19]) / os.sysconf('SC_CLK_TCK')
    except Exception:
        return time.time()


def load():
    if not CFG.exists():
        sys.exit(f'cost guard not initialised: run cost_guard.py init (missing {CFG})')
    return json.loads(CFG.read_text())


def spent(c, now=None):
    h = ((now or time.time()) - c['t0']) / 3600.0
    return h, h * (c['usd_per_h'] + c['storage_usd_per_h'])


def line(c, label='', est_h=None):
    h, usd = spent(c)
    s = (f'COST {label}: {h:.2f} pod-h elapsed, ~${usd:.2f} spent of ${c["budget_usd"]:.2f} budget '
         f'(${c["usd_per_h"]:.3f}/h GPU + ${c["storage_usd_per_h"]:.3f}/h disk; rates are guesses until set)')
    if est_h is not None:
        s += f'; next step est {est_h:.2f} h (~${est_h * (c["usd_per_h"] + c["storage_usd_per_h"]):.2f})'
    return s


def log(msg):
    STATE.mkdir(parents=True, exist_ok=True)
    m = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
    print(m, flush=True)
    with open(STATUS, 'a') as f:
        f.write(m + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('init')
    i.add_argument('--budget', type=float, default=float(os.environ.get('BUDGET_USD', 25)))
    i.add_argument('--rate', type=float, default=float(os.environ.get('RATE_USD_H', 0.34)))
    i.add_argument('--storage-gb', type=float, default=float(os.environ.get('STORAGE_GB', 250)))
    i.add_argument('--reserve', type=float, default=float(os.environ.get('RESERVE_USD', 0.30)),
                   help='kept back for syncing results and stopping cleanly')
    i.add_argument('--t0', type=float, default=None)
    i.add_argument('--force-rates', action='store_true', help='update budget/rates of an existing config (t0 kept)')
    c = sub.add_parser('check'); c.add_argument('label'); c.add_argument('--est-h', type=float, default=None)
    sub.add_parser('status')
    w = sub.add_parser('watch'); w.add_argument('pgid', type=int); w.add_argument('--every', type=float, default=60)
    a = ap.parse_args()
    if a.cmd == 'init':
        STATE.mkdir(parents=True, exist_ok=True)
        new = dict(budget_usd=a.budget, usd_per_h=a.rate, storage_usd_per_h=a.storage_gb * STORAGE_USD_PER_GB_MONTH / 730.0,
                   reserve_usd=a.reserve)
        if CFG.exists():
            old = json.loads(CFG.read_text())
            if a.force_rates:
                old.update(new)
                if a.t0:
                    old['t0'] = a.t0
                CFG.write_text(json.dumps(old, indent=1))
                log('cost guard rates updated: ' + json.dumps(old))
            print(line(json.loads(CFG.read_text()), 'init (existing config kept)'))
            return 0
        new['t0'] = a.t0 or pid1_start_epoch()
        new['t0_iso'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(new['t0']))
        CFG.write_text(json.dumps(new, indent=1))
        log('cost guard initialised: ' + json.dumps(new))
        return 0
    cfg = load()
    lim = cfg['budget_usd'] - cfg.get('reserve_usd', 0.0)
    rate = cfg['usd_per_h'] + cfg['storage_usd_per_h']
    if a.cmd == 'status':
        print(line(cfg, 'status')); return 0
    if a.cmd == 'check':
        h, usd = spent(cfg)
        log(line(cfg, a.label, a.est_h))
        if usd >= lim:
            log(f'BUDGET STOP before {a.label}: ~${usd:.2f} spent >= ${lim:.2f} (budget minus reserve). '
                f'Sync results back, then stop the pod.')
            return 3
        if a.est_h is not None and usd + a.est_h * rate > lim:
            log(f'BUDGET HOLD: {a.label} est ~${a.est_h * rate:.2f} would take spend to ~${usd + a.est_h * rate:.2f} '
                f'> ${lim:.2f}. Not started. Raise the budget (init --budget X --force-rates) or skip it.')
            return 4
        return 0
    if a.cmd == 'watch':
        while True:
            try:
                os.killpg(a.pgid, 0)
            except ProcessLookupError:
                return 0
            cfg = load(); lim = cfg['budget_usd'] - cfg.get('reserve_usd', 0.0)
            h, usd = spent(cfg)
            if usd >= lim:
                log(f'BUDGET STOP (watchdog): ~${usd:.2f} >= ${lim:.2f}; sending SIGTERM to job group {a.pgid}. '
                    f'Results so far are kept. Sync back, then stop the pod.')
                try:
                    os.killpg(a.pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                # quiet_super starts fit_spiral.py in its own session: stop it too (its autosave is kept)
                import subprocess
                subprocess.run(['pkill', '-TERM', '-f', 'fit_spiral.py'], check=False)
                return 3
            time.sleep(a.every)


if __name__ == '__main__':
    sys.exit(main())
