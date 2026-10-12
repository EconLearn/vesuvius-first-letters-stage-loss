#!/usr/bin/env python
"""Decide the winding sense (CW / ACW) of a band whose catalogue sense is unknown, from one fit per sense.

jobs/job3_quiet.sh fits such a band twice, with the scroll spec variants that umb_smooth.py wrote
(spiral-scroll.CW.json / spiral-scroll.ACW.json), into the run tags <tag>_cw and <tag>_acw. This script then decides:

RULE
  1. Both fits must have finished: <tag>_<sense>/checkpoint_fitted.ckpt with completed_iterations >= --steps and a log
     fit_<tag>_<sense>.log with logged losses ('step N: loss = X, ...' lines, one per 200 iterations). Otherwise REFUSE
     ("crashed or incomplete").
  2. Score of a sense = mean total loss over the logged steps N > steps - --window (default 500 iterations; at least the
     last --min-points = 3 logged values, so the default window of a 10000-step fit averages steps 9400, 9600, 9800).
     A step logged twice (a resumed fit) counts once, with its latest value. Non-finite losses REFUSE.
  3. Winner = the sense with the LOWER score.
  4. REFUSE as ambiguous when |CW - ACW| / min(CW, ACW) < --min-rel-diff (default 0.02 = 2 %).
  A forced mode (--mode cw|acw, i.e. SENSE_MODE=cw|acw) fits and checks only that sense and takes it without comparing.
  The rule is a heuristic (the wrong sense should fit the normals / tracks worse); it has not been validated here on a
  scroll with a known sense, so check it on one before trusting a decision.

OUTPUT on a decision: $dataset/sense_decision.json (both scores, rule, winner), and the winner becomes the canonical run:
  $QUIET_RUNS/<tag>/checkpoint_fitted.ckpt  (symlink to the winner's checkpoint; a copy where symlinks are unavailable),
  $QUIET_WORK/logs/fit_<tag>.log            (copy of the winner's log, only if absent),
  $dataset/spiral-scroll.json               (the winner's variant; the sense-less original is kept as
                                             spiral-scroll.nosense.json),
so flatten / screen proceed unchanged. On a refusal the same JSON records why (decision = refused) and the exit code is 6.

KNOWN-ANSWER CHECK (SENSE_VALIDATE=1 in jobs/job3_quiet.sh): with --expect SENSE --no-canonical --out FILE the same rule is
applied to the two fits of a band whose catalogue sense is known (quiet/sense_validate.py makes the variants); nothing is
canonicalised or overwritten, and the exit code is 7 if the rule chooses the other sense (6 if it refuses, as above).

usage: pick_sense.py --tag TAG --dataset DATASET_DIR [--steps 10000] [--mode both|cw|acw] [--window 500]
                     [--min-points 3] [--min-rel-diff 0.02] [--expect CW|ACW] [--no-canonical] [--out FILE]
env:   QUIET_WORK (default $VC_WORK/quiet), QUIET_RUNS (default $QUIET_WORK/runs); SENSE_WINDOW / SENSE_MIN_REL_DIFF
       set the defaults of --window / --min-rel-diff.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

LOSS = re.compile(r'^step (\d+): loss = ([^,\s]+)')
RULE = ('winner = lower mean total loss over the logged steps N > steps - window (at least the last min_points logged '
        'values); refuse if a fit crashed / is incomplete / has non-finite losses, or if |CW - ACW| / min(CW, ACW) < '
        'min_rel_diff (ambiguous)')
QW = Path(os.environ.get('QUIET_WORK', os.path.join(_P.WORK, 'quiet')))
RUNS = Path(os.environ.get('QUIET_RUNS', str(QW / 'runs')))
LOGS = QW / 'logs'


class Refused(Exception):
    pass


def read_losses(log):
    """{step: loss} from a fit log; later lines of the same step (resumed fit) override earlier ones."""
    if not Path(log).is_file():
        raise Refused(f'no fit log {log}')
    out = {}
    for line in Path(log).read_text(errors='replace').replace('\r', '\n').split('\n'):
        m = LOSS.match(line.strip())
        if m:
            try:
                out[int(m.group(1))] = float(m.group(2))
            except ValueError:
                out[int(m.group(1))] = float('nan')
    return out


def window_mean(losses, steps, window, min_points, what):
    pts = sorted(losses.items())
    if len(pts) < min_points:
        raise Refused(f'{what}: only {len(pts)} logged loss values (need >= {min_points}); the fit did not run properly')
    use = [(s, v) for s, v in pts if s > steps - window]
    if len(use) < min_points:
        use = pts[-min_points:]
    if not all(math.isfinite(v) for _, v in use):
        raise Refused(f'{what}: non-finite loss in the scoring window (steps {[s for s, _ in use]})')
    return sum(v for _, v in use) / len(use), [s for s, _ in use], len(pts)


def ckpt_iterations(path):
    try:
        import torch
        return int(torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)['completed_iterations'])
    except Exception:
        return None


def has_sense(path):
    try:
        return json.loads(Path(path).read_text()).get('spiral_outward_sense')
    except Exception:
        return None


def canonicalise(tag, winner, ds):
    """Make the winner's run the canonical <tag> run and its spec variant the dataset's spiral-scroll.json."""
    w = winner.lower()
    src = RUNS / f'{tag}_{w}' / 'checkpoint_fitted.ckpt'
    run = RUNS / tag
    dst = run / 'checkpoint_fitted.ckpt'
    run.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() != src.resolve():
            raise Refused(f'{dst} already exists and is not the {winner} run; not overwriting')
    else:
        try:
            os.symlink(os.path.relpath(src, run), dst)
        except OSError:
            shutil.copy2(src, dst)
    LOGS.mkdir(parents=True, exist_ok=True)
    clog = LOGS / f'fit_{tag}.log'
    if not clog.exists():
        shutil.copyfile(LOGS / f'fit_{tag}_{w}.log', clog)
    main, var, nosense = ds / 'spiral-scroll.json', ds / f'spiral-scroll.{winner}.json', ds / 'spiral-scroll.nosense.json'
    if not var.is_file():
        raise Refused(f'missing scroll spec variant {var}')
    cur = has_sense(main) if main.is_file() else None
    if cur and cur != winner:
        raise Refused(f'{main} already holds sense {cur}; not overwriting it with {winner}')
    if main.is_file() and not cur and not nosense.exists():
        shutil.copyfile(main, nosense)
    shutil.copyfile(var, main)
    return dict(checkpoint=str(dst), log=str(clog), spec=str(main), spec_backup=str(nosense) if nosense.exists() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--mode', default='both', choices=['both', 'cw', 'acw'])
    ap.add_argument('--window', type=int, default=int(os.environ.get('SENSE_WINDOW', '500')))
    ap.add_argument('--min-points', type=int, default=3)
    ap.add_argument('--min-rel-diff', type=float, default=float(os.environ.get('SENSE_MIN_REL_DIFF', '0.02')))
    ap.add_argument('--expect', choices=['CW', 'ACW'], help='known-answer check: exit 7 unless the rule chooses this sense (needs --mode both)')
    ap.add_argument('--no-canonical', action='store_true', help='decide and record only; leave run directories and specs alone')
    ap.add_argument('--out', default=None, help='decision JSON (default <dataset>/sense_decision.json)')
    a = ap.parse_args()
    ds = Path(a.dataset)
    senses = {'both': ['CW', 'ACW'], 'cw': ['CW'], 'acw': ['ACW']}[a.mode]
    rec = dict(dataset=ds.name, tag=a.tag, mode=a.mode, rule=RULE, steps=a.steps, window_iterations=a.window,
               min_points=a.min_points, min_rel_diff=a.min_rel_diff, senses={}, winner=None, rel_diff=None,
               decision=None, reason=None, decided_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    out = Path(a.out) if a.out else ds / 'sense_decision.json'
    try:
        if a.expect and a.mode != 'both':
            raise Refused('--expect needs --mode both')
        for sn in senses:
            tg = f'{a.tag}_{sn.lower()}'
            ck = RUNS / tg / 'checkpoint_fitted.ckpt'
            it = ckpt_iterations(ck) if ck.exists() else None
            if it is None or it < a.steps:
                raise Refused(f'{sn} fit crashed or is incomplete: {ck} has '
                              f'{"no readable checkpoint" if it is None else f"{it} of {a.steps} iterations"}')
            mean, used, n = window_mean(read_losses(LOGS / f'fit_{tg}.log'), a.steps, a.window, a.min_points, f'{sn} fit')
            rec['senses'][sn] = dict(tag=tg, mean_loss=mean, steps_used=used, n_logged=n, checkpoint=str(ck))
        if a.mode == 'both':
            cw, acw = rec['senses']['CW']['mean_loss'], rec['senses']['ACW']['mean_loss']
            lo = min(cw, acw)
            rec['rel_diff'] = abs(cw - acw) / lo if lo > 0 else float('inf')
            if rec['rel_diff'] < a.min_rel_diff:
                raise Refused(f'ambiguous: CW {cw:.1f} vs ACW {acw:.1f} (relative difference {100 * rec["rel_diff"]:.1f} % '
                              f'< {100 * a.min_rel_diff:.1f} %); set SENSE_MODE=cw or SENSE_MODE=acw to force one')
            rec['winner'] = 'CW' if cw < acw else 'ACW'
            rec['decision'] = 'winner'
        else:
            rec['winner'] = senses[0]
            rec['decision'] = 'forced'
        if a.expect:
            rec['expect'] = a.expect
            if rec['winner'] != a.expect:
                rec['decision'] = 'validation_failed'
                rec['reason'] = f'the rule chose {rec["winner"]} but the catalogue sense is {a.expect}'
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=1))
                sc = ' '.join(f'{k}={v["mean_loss"]:.1f}' for k, v in rec['senses'].items())
                print(f'SENSE VALIDATION FAILED {a.tag}: the rule chose {rec["winner"]} but the catalogue sense is {a.expect} '
                      f'(mean loss {sc}, rel diff {100 * rec["rel_diff"]:.1f} %)')
                return 7
            rec['validated'] = True
        if not a.no_canonical:
            rec['canonical'] = canonicalise(a.tag, rec['winner'], ds)
    except Refused as e:
        rec['decision'] = 'refused'; rec['reason'] = str(e); rec['winner'] = None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1))
        print(f'SENSE FAIL {a.tag}: {e}')
        return 6
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    sc = ' '.join(f'{k}={v["mean_loss"]:.1f}' for k, v in rec['senses'].items())
    if a.expect:
        print(f'SENSE VALIDATION PASSED {a.tag}: the rule chose {rec["winner"]} = the catalogue sense (mean loss {sc}, '
              f'rel diff {100 * rec["rel_diff"]:.1f} %) -> {out}')
        return 0
    print(f'SENSE DECISION {a.tag}: winner={rec["winner"]} ({rec["decision"]}, mode {a.mode}) mean loss {sc}'
          + (f' rel diff {100 * rec["rel_diff"]:.1f} %' if rec['rel_diff'] is not None else '') + f' -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
