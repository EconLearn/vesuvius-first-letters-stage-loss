#!/usr/bin/env python
"""JOB 1 report: fit speed (stop rule), villa inference speed, render/CT speed, GPU utilisation -> timing.json and
re-projected per-job hours and dollars. Every projection is labelled a GUESS built from the measured rates plus the
stated area assumptions.

Fit rate: PROGRESS lines 'N/TOTAL iterations ... elapsed E' of the LAST start block of the fit log. r_pre = slope of
N over elapsed before loss_start_track_dt, r_post = after it. The full recipe (10k steps, track dt from 7000) runs at
r_eff = 10000 / (7000 / r_pre + 3000 / r_post). STOP RULE (pre-set): proceed only if r_eff >= 3.0 it/s.
RTX 4060 reference: the same parser on three earlier 10k tracks-on fit logs (stored in REF_4060 below, re-measurable
with --ref-log).

usage: timing_report.py --fit-log LOG [--track-start 250] --render R.zarr --render-json J --sweep LABEL=DIR ...
                        [--gpu LABEL=CSV ...] [--rate 0.34] --out timing.json
       timing_report.py --ref-log $QUIET_WORK/logs/fit_<tag>.log --track-start 7000   (parser check)
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ENS = Path(os.environ.get('ENS_LANE', str(Path(__file__).resolve().parents[2] / 'ensemble')))    # tools/ensemble
REF_4060 = dict(source='RTX 4060 workstation, measured by this parser on three 10k tracks-on fit logs (dt from 7000)',
                ref_fit_1=dict(r_pre=2.12, r_post=1.68, r_eff=1.96), ref_fit_2=dict(r_pre=2.11, r_post=1.98, r_eff=2.07),
                ref_fit_3=dict(r_pre=1.95, r_post=1.59, r_eff=1.83))
VILLA_4060_S_PER_MPX = (11.1, 38.3)                        # earlier cost-model anchors (one order, one depth)
STOP_IT_S = 3.0
PROG = re.compile(r'Optimizing\D+?([\d,]+)/([\d,]+) iterations.*?elapsed\s+((?:\d+h\s*)?(?:\d+m\s*)?(?:[\d.]+s)?)')


def secs(e):
    t = 0.0
    for v, u in re.findall(r'([\d.]+)([hms])', e):
        t += float(v) * dict(h=3600, m=60, s=1)[u]
    return t


def parse_fit(path, track_start):
    txt = Path(path).read_text(errors='replace')
    blk = txt.split('==== START')[-1]
    pts = []
    for line in blk.replace('\r', '\n').split('\n'):
        m = PROG.search(line)
        if m:
            pts.append((int(m.group(1).replace(',', '')), secs(m.group(3)), int(m.group(2).replace(',', ''))))
    if len(pts) < 4:
        return dict(error=f'only {len(pts)} progress points in {path}')
    N = np.array([p[0] for p in pts], float); T = np.array([p[1] for p in pts], float); total = pts[-1][2]

    def slope(lo, hi):
        k = (N >= lo) & (N <= hi)
        if k.sum() < 3 or np.ptp(T[k]) <= 0:
            return None
        return float(np.polyfit(T[k], N[k], 1)[0])
    ts = track_start
    r_pre = slope(max(30, 0.1 * ts), ts)
    r_post = slope(ts + 0.1 * (total - ts), total) if total > ts else None
    r_eff = None
    if r_pre and r_post:
        r_eff = 10000.0 / (7000.0 / r_pre + 3000.0 / r_post)
    elif r_pre:
        r_eff = r_pre
    return dict(log=str(path), total=total, track_start=ts, n_points=len(pts), last_iter=int(N[-1]),
                elapsed_s=float(T[-1]), r_pre=r_pre, r_post=r_post, r_eff=r_eff)


def valid_px(render):
    sys.path.insert(0, str(ENS))
    from ens import common as C
    arr, at = C.open_render(render)
    return int(C.valid_mask_all_layers(arr, k_range=at['ensemble_render'].get('rendered_k')).sum())


def parse_sweep(d, nvalid):
    rows = []
    tf = Path(d) / 'timings.tsv'
    if not tf.exists():
        return rows
    for r in csv.reader(open(tf), delimiter='\t'):
        if len(r) < 7:
            continue
        rows.append(dict(model=r[0], d=int(r[1]), secs=float(r[4]), canvas_px=int(r[5]), patches=int(r[6])))
    for r in rows:
        r['s_per_valid_mpx'] = r['secs'] / max(nvalid / 1e6, 1e-9)
        r['patches_per_s'] = r['patches'] / max(r['secs'], 1e-9)
    return rows


def gpu_stats(path):
    u, m = [], []
    for line in open(path):
        p = [x.strip() for x in line.split(',')]
        try:
            u.append(float(p[-2])); m.append(float(p[-1]))
        except (ValueError, IndexError):
            continue
    if not u:
        return None
    return dict(samples=len(u), util_mean=round(float(np.mean(u)), 1), util_p90=round(float(np.percentile(u, 90)), 1),
                mem_max_mib=max(m))


def project(fit, villa, render, rate, band_mpx=1034.0, top=0.10, ctx=1.4, s2_mpx=11.0):
    """Hours per job from the measured rates. GUESSES: area numbers are a 1000-slice PHerc0343 band (1034 valid Mpx),
    his stage-2 canvas (17.2 Mpx x 64 % valid), top 10 % x 1.4 patch context; 1500-slice fit = 1.5x
    the 1000-slice fit time (per-step sample counts scale with the z range)."""
    out = {}
    r = fit.get('r_eff') or 1.0
    fit_h = 10000 / r / 3600 + 0.1
    dn = villa.get('dnative', 15.3); three = sum(villa.get(m, dn) for m in ('ink9_s42', 'ink9_s43', 'dnative'))
    s_r = render.get('s_per_mpx_layer', 0.5)
    s2_sweep = s2_mpx * (7 * 2 + 3) * three / 3600
    s2_render = s2_mpx * 66 * s_r / 3600
    out['job2_stage2_343'] = dict(fit_1500_slices_h=round(1.5 * fit_h, 2), flatten_h=0.1, render_h=round(s2_render, 2),
                                  sweeps_h=round(s2_sweep, 2), post_h=0.2)
    screen_gpu = band_mpx * 3 * dn / 3600; screen_cpu = band_mpx * 21 * s_r / 3600
    ens_gpu = band_mpx * top * ctx * 7 * three / 3600; ens_cpu = band_mpx * top * ctx * 29 * s_r / 3600
    out['job3_per_scroll'] = dict(fit_h=round(fit_h, 2), flatten_h=0.1, screen_gpu_h=round(screen_gpu, 2),
                                  screen_render_cpu_h=round(screen_cpu, 2), ens_gpu_h=round(ens_gpu, 2),
                                  ens_render_cpu_h=round(ens_cpu, 2), draw_h=0.2)
    for k, v in out.items():
        if k == 'job3_per_scroll':
            tot = v['fit_h'] + v['flatten_h'] + max(v['screen_gpu_h'], v['screen_render_cpu_h']) + v['ens_gpu_h'] + \
                v['ens_render_cpu_h'] + v['draw_h']
        else:
            tot = sum(x for x in v.values())
        v['total_h'] = round(tot, 2); v['usd'] = round(tot * rate, 2)
    j3 = out['job3_per_scroll']
    out['job3_seven_scrolls'] = dict(total_h=round(7 * j3['total_h'], 1), usd=round(7 * j3['usd'], 1))
    out['assumptions'] = project.__doc__
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit-log'); ap.add_argument('--track-start', type=int, default=250)
    ap.add_argument('--ref-log'); ap.add_argument('--render'); ap.add_argument('--render-json')
    ap.add_argument('--sweep', action='append', default=[]); ap.add_argument('--gpu', action='append', default=[])
    ap.add_argument('--rate', type=float, default=0.34); ap.add_argument('--stop-it-s', type=float, default=STOP_IT_S)
    ap.add_argument('--out'); ap.add_argument('--valid-px', type=int, help='instead of --render: valid px of the swept canvas')
    a = ap.parse_args()
    if a.ref_log and not a.fit_log:
        r = parse_fit(a.ref_log, a.track_start)
        print(json.dumps(r, indent=1)); return 0
    res = dict(ref_4060=REF_4060, villa_4060_s_per_mpx=VILLA_4060_S_PER_MPX)
    if a.ref_log:
        res['ref_4060'] = dict(REF_4060, remeasured={k: v for k, v in parse_fit(a.ref_log, 7000).items() if k.startswith('r_')})
    fit = parse_fit(a.fit_log, a.track_start) if a.fit_log else dict(error='no fit log')
    res['fit'] = fit
    ok = bool(fit.get('r_eff') and fit['r_eff'] >= a.stop_it_s)
    res['stop_rule'] = dict(rule=f'proceed only if the fit runs at >= {a.stop_it_s} it/s (r_eff of the 10k recipe)',
                            r_eff=fit.get('r_eff'), passed=ok)
    nvalid = valid_px(a.render) if a.render else (a.valid_px or 0)
    res['timing_canvas_valid_px'] = nvalid
    detail, best = [], {}
    for spec in a.sweep:
        lab, d = spec.split('=', 1)
        rows = parse_sweep(d, nvalid)
        for r in rows:
            r['label'] = lab
        detail += rows
        for m in set(r['model'] for r in rows):
            v = float(np.median([r['s_per_valid_mpx'] for r in rows if r['model'] == m]))
            if m not in best or v < best[m][0]:
                best[m] = (v, lab)
    res['villa_detail'] = detail
    res['villa_s_per_valid_mpx'] = {m: round(v[0], 2) for m, v in best.items()}
    res['villa_best_setting'] = {m: v[1] for m, v in best.items()}
    res['villa_speedup_vs_4060_anchor'] = {m: [round(VILLA_4060_S_PER_MPX[0] / v[0], 2), round(VILLA_4060_S_PER_MPX[1] / v[0], 2)]
                                           for m, v in best.items()}
    rj = json.loads(Path(a.render_json).read_text()) if a.render_json and Path(a.render_json).exists() else {}
    if rj and nvalid:
        rj['s_per_mpx_layer'] = rj['secs'] / (nvalid / 1e6 * rj['layers'])
        rj['fetch_mb_per_s'] = rj.get('fetched_bytes', 0) / 1e6 / max(rj['secs'], 1e-9)
    res['render'] = rj
    res['gpu'] = {}
    for spec in a.gpu:
        lab, f = spec.split('=', 1)
        if Path(f).exists():
            res['gpu'][lab] = gpu_stats(f)
    rec = []
    fu = (res['gpu'].get('fit') or {}).get('util_mean')
    if fu is not None and fu < 60:
        rec.append(f'fit GPU utilisation {fu}% < 60%: OVERLAP=1 (next fit shares the card with the sweeps) likely raises throughput')
    su = (res['gpu'].get('sweep') or {}).get('util_mean')
    if su is not None and su < 50:
        rec.append(f'villa sweep GPU utilisation {su}% < 50%: inference is CPU/loader-bound (robust-MAD normalisation on '
                   f'6 vCPU); a GPU-side normalisation port would be the biggest speed lever (not done: it changes validated code)')
    res['recommendations'] = rec
    res['projections_GUESS'] = project(fit, res['villa_s_per_valid_mpx'], rj, a.rate)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"FIT: r_pre {fit.get('r_pre')} r_post {fit.get('r_post')} r_eff {fit.get('r_eff')} it/s -> "
          f"STOP RULE {'PASS' if ok else 'FAIL'} (>= {a.stop_it_s})")
    print('VILLA s per valid Mpx per depth:', res['villa_s_per_valid_mpx'], 'best:', res['villa_best_setting'])
    if rj:
        print(f"RENDER: {rj.get('s_per_mpx_layer', float('nan')):.3f} s per valid Mpx-layer, fetch {rj.get('fetch_mb_per_s', 0):.0f} MB/s")
    print('GPU:', json.dumps(res['gpu']))
    for k, v in res['projections_GUESS'].items():
        if k != 'assumptions':
            print('PROJECTION (guess)', k, json.dumps(v))
    for r in rec:
        print('RECOMMEND', r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
