"""Automatic umbilici (scroll axes) for First Letters scrolls that have none, from the published m7 normal grids.

Estimator: VC3D align_and_extract_umbilicus (villa volume-cartographer/core/src/normalgridtools.cpp), compiled
stand-alone (harness/), three variants:
  unchanged  : $UMB_HARNESS 0 SEED files...                     (VC3D as published)
  midpoint   : $UMB_HARNESS 0.5 SEED files...                   (whiten by midpoint covariance, alpha 0.5)
  normal     : UNCRUSH_N=1 $UMB_HARNESS_N 1.5 SEED files...     (whiten by normal orientation tensor, beta 1.5) PRIMARY

usage (WSL python):
  umbkit.py run  [UNIT ...]   download grids to /dev/shm, run 3 methods x 9 seeds, write work/<UNIT>.raw.json
  umbkit.py build [UNIT ...]  flags, smoothing, umbilicus.json, umbilicus_all_methods.json, qa.png, summary.json
"""
import json, os, sys, time, struct, hashlib, shutil, subprocess, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

W = Path(os.environ.get('UMBKIT_DIR', Path(__file__).resolve().parents[1])); WORK = W / 'work'; QAG = W / 'qa_grids'
SHM = Path('/dev/shm/umbkit'); B = 'vesuvius-challenge-open-data'
H0, HN = os.environ.get('UMB_HARNESS', '/tmp/umbkit_build/vc3d_harness'), os.environ.get('UMB_HARNESS_N', '/tmp/umbkit_build/vc3d_harness_n')
SEEDS = list(range(1, 10)); STEP_FULL = 200; MIN_SEG = 500
SPREAD_MM, SPLIT_MM, WIN, REJ_MM = 2.0, 3.0, 5, 3.0
METHODS = {  # name -> (binary, strength, env UNCRUSH_N, label)
    'unchanged': (H0, '0', False, 'VC3D unchanged (alpha 0)'),
    'midpoint': (H0, '0.5', False, 'midpoint un-crush (alpha 0.5)'),
    'normal': (HN, '1.5', True, 'normal un-crush (beta 1.5)'),
}
PRIMARY = os.environ.get('UMB_PRIMARY', 'normal')
KIT = Path(os.environ.get('VC_SPIRAL_DATA', W / 'no_kit_route'))  # read only: the cloud kit's own umbilici, for the agreement column (optional)


def units():
    inv = json.loads((W / 's3_inventory.json').read_text()); vm = json.loads((W / 'volume_metadata.json').read_text())
    out = {}
    for s, r in inv.items():
        if not r.get('top_level') or not r.get('m7_normal_grids'):
            continue
        for ng in r['m7_normal_grids']:
            stamp = ng.split('-')[0]; vol = [v for v in r['zarr'] if v.startswith(stamp)]
            if len(vol) != 1:
                print(s, ng, 'no unique volume for stamp', stamp); continue
            vol = vol[0]; za = json.loads(r['zarr'][vol]['0/.zarray']); Z, Y, X = za['shape']
            nxy = r['ng'][ng]['xy_n']
            if nxy == Z: f = 1
            elif nxy in (Z // 4, -(-Z // 4)): f = 4
            else:
                print(s, ng, 'xy count', nxy, 'matches neither', Z, 'nor', Z / 4); continue
            acq = vm[f'{s}|{vol}'].get('scan', {}).get('tomo', {}).get('acquisition', {})
            vox = float(vol.split('-')[1].replace('um', ''))
            pix = acq.get('detector', {}).get('samplePixelSize')
            name = s if vol.split('-')[1] in ('8.640um', '9.362um') else f'{s}/alt_volume_{stamp}-{vol.split("-")[1]}'
            out[name] = dict(scroll=s, vol=vol, ng=f'{B}/{s}/representations/predictions/surfaces/{ng}', ng_name=ng, f=f,
                             shape_zyx=[Z, Y, X], vox_um=vox, sample_pixel_mm=pix, scan=acq.get('scanRadix'), xy_n=nxy)
    return out


def ukey(name): return name.replace('/', '__')


def header(b):
    h = struct.unpack('>4s12I', b[:52]); return dict(w=h[4], h=h[5], step=h[6], n_paths=h[8], off=h[10])


def paths(b):
    hd = header(b); o = hd['off']; P = []
    for _ in range(hd['n_paths']):
        x, y = struct.unpack('>ii', b[o:o + 8]); n = struct.unpack('>I', b[o + 8:o + 12])[0]; o += 12
        d = np.frombuffer(b[o:o + n], dtype=np.int8).reshape(-1, 2).astype(np.int64); o += n
        P.append(np.vstack([[x, y], [x, y] + np.cumsum(d, 0)]))
    return P


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetch(fs, key, dst):
    for att in range(6):
        try:
            b = fs.cat_file(key); dst.write_bytes(b); return len(b)
        except Exception as e:
            err = e; time.sleep(4 * (att + 1))
    raise err


def run_unit(name, U):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True); f = U['f']; step = STEP_FULL // f
    sizes = json.loads((W / 'xy_sizes.json').read_text())[f"{U['scroll']}|{U['ng_name']}"]
    nz = sorted(int(k) for k, v in sizes.items() if v > 52)
    g0, g1 = nz[0], nz[-1]; zs = [z for z in range(-(-g0 // step) * step, g1 + 1, step)]
    d = SHM / ukey(name); d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        nb = list(ex.map(lambda z: fetch(fs, f"{U['ng']}/xy/{z:06d}.grid", d / f'{z:06d}.grid') if not (d / f'{z:06d}.grid').exists() else (d / f'{z:06d}.grid').stat().st_size, zs))
    seg = {}; dims = set()
    for z in zs:
        hd = header((d / f'{z:06d}.grid').read_bytes()[:52]); seg[z] = hd['n_paths']; dims.add((hd['w'], hd['h']))
    used = [z for z in zs if seg[z] >= MIN_SEG]
    print(f'{name}: {len(zs)} slices (grid z {zs[0]}..{zs[-1]} step {step}), {sum(nb)/1e6:.1f} MB in {time.time()-t0:.0f} s, '
          f'{len(used)} with >= {MIN_SEG} segments, grid dims {dims}', flush=True)
    files = [str(d / f'{z:06d}.grid') for z in used]

    def job(ms):
        m, sd = ms; binp, a, n, _ = METHODS[m]
        env = dict(os.environ); env.pop('UNCRUSH_N', None)
        if n: env['UNCRUSH_N'] = '1'
        r = subprocess.run(['nice', '-n', '10', binp, a, str(sd)] + files, capture_output=True, text=True, env=env)
        L = [list(map(float, l.split())) for l in r.stdout.split('\n') if l.strip()]
        if len(L) != len(files): raise RuntimeError(f'{name} {m} seed {sd}: {len(L)} lines for {len(files)} files; rc {r.returncode} {r.stderr[:300]}')
        return m, sd, L
    t1 = time.time(); est = {m: {} for m in METHODS}
    with ThreadPoolExecutor(4) as ex:  # at most 4 harness processes at once
        for m, sd, L in ex.map(job, [(m, sd) for m in METHODS for sd in SEEDS]):
            est[m][sd] = L
    print(f'{name}: harness 3 methods x {len(SEEDS)} seeds on {len(files)} slices in {time.time()-t1:.0f} s', flush=True)
    slices = []
    for i, z in enumerate(zs):
        r = dict(z_grid=z, z=z * f, n_segments=seg[z], bytes=nb[i], used=z in used)
        if z in used:
            j = used.index(z); r['est'] = {m: [est[m][sd][j] for sd in SEEDS] for m in METHODS}
        slices.append(r)
    # QA slices: 6 used slices nearest 10..90 % of the used z extent
    qz = sorted({min(used, key=lambda u: abs(u - (used[0] + q * (used[-1] - used[0])))) for q in np.linspace(0.1, 0.9, 6)})
    qd = QAG / ukey(name); qd.mkdir(parents=True, exist_ok=True)
    for z in qz: shutil.copy(d / f'{z:06d}.grid', qd / f'{z:06d}.grid')
    raw = dict(unit=name, U=U, step_grid=step, slices=slices, qa_z_grid=qz, seeds=SEEDS, min_segments=MIN_SEG,
               harness_sha256={H0: sha(H0), HN: sha(HN)}, run_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    WORK.mkdir(exist_ok=True); (WORK / f'{ukey(name)}.raw.json').write_text(json.dumps(raw))
    shutil.rmtree(d)  # free /dev/shm


# ---------------------------------------------------------------- build
def seed_stats(E, mm):
    E = np.array(E, float); ok = np.all(np.isfinite(E), 1); E = E[ok]
    if len(E) < 5: return None
    med = np.median(E, 0); dist = np.linalg.norm(E - med, axis=1) * mm; spread = float(np.median(dist))
    from scipy.cluster.hierarchy import linkage, fcluster
    lab = fcluster(linkage(E, 'complete'), 2, 'maxclust'); sizes = [int(np.sum(lab == k)) for k in (1, 2)]
    sep = float(np.linalg.norm(np.median(E[lab == 1], 0) - np.median(E[lab == 2], 0)) * mm) if min(sizes) > 0 else 0.0
    split = min(sizes) >= 2 and sep > SPLIT_MM
    return dict(median=med.tolist(), spread_mm=spread, max_dev_mm=float(dist.max()), n_finite=int(len(E)),
                cluster_sizes=sizes, cluster_sep_mm=sep, split=bool(split), unsure=bool(spread > SPREAD_MM or split))


def runmed(P, w=WIN):
    h = w // 2; return np.array([np.median(P[max(0, i - h):i + h + 1], 0) for i in range(len(P))])


TRIM = os.environ.get('UMB_TRIM', '1') == '1'; TRIM_RUN = 3; END_STRETCH = 4; END_JUMP_MM = 5.0


def smooth(z, P, mm, unsure):
    """Running median over WIN slices; reject points > REJ_MM from it. End trim (UMB_TRIM=1, default): control points start
    at the first and end at the last run of TRIM_RUN consecutive slices that are kept and not UNSURE, so the scan ends (where the
    estimator locks onto container walls or mask edges and the truncated median window cannot outvote them) give no points.
    Control points = running median over WIN of the kept points inside that range."""
    rm = runmed(P); dev = np.linalg.norm(P - rm, axis=1) * mm; keep = dev <= REJ_MM
    inr = np.ones(len(z), bool)
    if TRIM:
        ok = keep & ~np.asarray(unsure, bool); runs = [i for i in range(len(z) - TRIM_RUN + 1) if ok[i:i + TRIM_RUN].all()]
        if runs:
            inr[:runs[0]] = False; inr[runs[-1] + TRIM_RUN:] = False
        # consistent-but-wrong ends: an end stretch of <= END_STRETCH kept points separated from the rest by a jump
        # > END_JUMP_MM between consecutive kept raw points is trimmed too (the running median cannot outvote it)
        idx = np.flatnonzero(keep & inr)
        if len(idx) > 2 * END_STRETCH:
            jump = np.linalg.norm(np.diff(P[idx], axis=0), axis=1) * mm
            j0 = [k for k in range(END_STRETCH) if jump[k] > END_JUMP_MM]
            if j0: inr[:idx[j0[-1] + 1]] = False
            j1 = [k for k in range(len(jump) - END_STRETCH, len(jump)) if jump[k] > END_JUMP_MM]
            if j1: inr[idx[j1[0]] + 1:] = False
    sel = keep & inr; sm = runmed(P[sel]); return rm, dev, keep, inr, sel, sm


def kit_route(scroll):
    p = KIT / scroll / 'umbilicus.json'
    if not p.exists(): return None
    u = json.loads(p.read_text()); C = np.array(sorted([[c['z'], c['x'], c['y']] for c in u['control_points']]), float)
    ps = u.get('provenance', {}).get('per_slice', []); zps = [r['z'] for r in ps]
    sp = KIT / scroll / 'spiral-scroll.json'; spj = json.loads(sp.read_text()) if sp.exists() else {}
    return dict(C=C, z_band=[min(zps), max(zps)] if zps else [C[0, 0], C[-1, 0]], vox_um=spj.get('voxel_size_um'),
                shape_zyx=spj.get('base_shape_zyx'), method=u.get('provenance', {}).get('method'), inputs=u.get('provenance', {}).get('inputs'))


def build_unit(name):
    raw = json.loads((WORK / f'{ukey(name)}.raw.json').read_text()); U = raw['U']; f = U['f']
    mm = U['vox_um'] * f / 1000.0  # mm per grid pixel
    used = [s for s in raw['slices'] if s['used']]
    per = []
    for s in used:
        r = dict(z=s['z'], z_grid=s['z_grid'], n_segments=s['n_segments'])
        for m in METHODS:
            st = seed_stats(s['est'][m], mm)
            r[m] = None if st is None else dict(x=st['median'][0] * f, y=st['median'][1] * f, spread_mm=round(st['spread_mm'], 3),
                                                max_dev_mm=round(st['max_dev_mm'], 3), cluster_sizes=st['cluster_sizes'],
                                                cluster_sep_mm=round(st['cluster_sep_mm'], 3), split=st['split'], unsure=st['unsure'],
                                                n_finite=st['n_finite'], seeds_xy=[[a * f, b * f] for a, b in s['est'][m]])
        per.append(r)
    smoothed = {}
    for m in METHODS:
        rows = [r for r in per if r[m] is not None]
        z = np.array([r['z'] for r in rows], float); P = np.array([[r[m]['x'], r[m]['y']] for r in rows], float)
        rm, dev, keep, inr, sel, sm = smooth(z, P, U['vox_um'] / 1000.0, [r[m]['unsure'] for r in rows])
        for r, a, d_, k, t in zip(rows, rm, dev, keep, inr):
            r[m].update(running_median=[round(a[0], 1), round(a[1], 1)], dev_from_running_median_mm=round(float(d_), 3), kept=bool(k),
                        in_trimmed_range=bool(t))
        zt = z[inr]
        smoothed[m] = dict(z=z[sel].astype(int).tolist(), x=np.round(sm[:, 0], 1).tolist(), y=np.round(sm[:, 1], 1).tolist(),
                           n_kept=int(keep.sum()), n_rejected=int((~keep).sum()), rejected_z=z[~keep].astype(int).tolist(),
                           trimmed_z=z[~inr].astype(int).tolist(), trimmed_range=[int(zt[0]), int(zt[-1])] if len(zt) else None,
                           n_points=int(sel.sum()), unsure_z=[r['z'] for r in rows if r[m]['unsure']],
                           unsure_in_range=int(sum(1 for r, t in zip(rows, inr) if t and r[m]['unsure'])),
                           rejected_in_range=int(np.sum(inr & ~keep)))
    out_dir = W / name; out_dir.mkdir(parents=True, exist_ok=True)
    Z, Y, X = U['shape_zyx']; now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    tool = ('VC3D align_and_extract_umbilicus (villa volume-cartographer core/src/normalgridtools.cpp), compiled stand-alone '
            'against OpenCV stand-ins; un-crush option added (umbkit harness/)')
    base_md = dict(volume=U['vol'], volume_width=X, volume_height=Y, volume_slices=Z, voxelsize_um=U['vox_um'], scan=U['scan'],
                   source_annotation='automatic (no hand annotation); umbkit', tool=tool,
                   normal_grid=f"s3://{U['ng']}", grid_level=0 if f == 1 else 2, grid_to_fullres_factor=f,
                   coordinate_frame='x, y, z in full-resolution voxels of the volume (zarr level 0)',
                   seeds=SEEDS, seed_aggregation='coordinatewise median of the per-seed estimates',
                   slice_step=STEP_FULL, min_segments_per_slice=MIN_SEG,
                   smoothing=f'running median over {WIN} slices; points more than {REJ_MM} mm from it rejected; '
                             + (f'ends trimmed to the first/last run of {TRIM_RUN} consecutive kept, not-UNSURE slices, and an end stretch of '
                                f'<= {END_STRETCH} kept points separated from the rest by a jump > {END_JUMP_MM} mm is trimmed; ' if TRIM else 'no end trim; ')
                             + f'control points = running median over {WIN} of the kept points in that range',
                   unsure_rule=f'seed spread (median distance of the 9 estimates from their median) > {SPREAD_MM} mm, or the seeds '
                               f'split into two clusters (complete linkage, each >= 2 seeds) more than {SPLIT_MM} mm apart',
                   harness_sha256=raw['harness_sha256'], run_utc=raw['run_utc'], timestamp=now)
    mlab = METHODS[PRIMARY][3]; sm = smoothed[PRIMARY]
    md = dict(base_md, method=mlab, alpha=None if PRIMARY == 'normal' else float(METHODS[PRIMARY][1]),
              beta=1.5 if PRIMARY == 'normal' else None, total_points=len(sm['z']), unsure_z=sm['unsure_z'], rejected_z=sm['rejected_z'],
              trimmed_z=sm['trimmed_z'], z_range_of_control_points=[sm['z'][0], sm['z'][-1]] if sm['z'] else None,
              note='Automatic axis estimate, not a hand annotation. Check qa.png before use.')
    cp = [dict(x=x, y=y, z=z, score=100) for x, y, z in zip(sm['x'], sm['y'], sm['z'])]
    (out_dir / 'umbilicus.json').write_text(json.dumps(dict(control_points=cp, metadata=md), indent=1))
    kit = kit_route(U['scroll']) if f == 1 else None
    agree = None
    if kit is not None:
        agree = dict(kit_method=kit['method'], kit_inputs=kit['inputs'], kit_vox_um=kit['vox_um'], kit_shape_zyx=kit['shape_zyx'],
                     frame_matches_volume=(kit['shape_zyx'] == U['shape_zyx'] and abs((kit['vox_um'] or 0) - U['vox_um']) < 1e-3),
                     kit_z_band=kit['z_band'])
        C = kit['C']; vmm = U['vox_um'] / 1000.0
        for m in METHODS:
            s_ = smoothed[m]; zz = np.array(s_['z']); sel = (zz >= kit['z_band'][0]) & (zz <= kit['z_band'][1])
            if sel.sum() == 0:
                agree[m] = dict(n=0); continue
            kx = np.interp(zz[sel], C[:, 0], C[:, 1]); ky = np.interp(zz[sel], C[:, 0], C[:, 2])
            dd = np.hypot(np.array(s_['x'])[sel] - kx, np.array(s_['y'])[sel] - ky) * vmm
            agree[m] = dict(n=int(sel.sum()), z=zz[sel].tolist(), dist_mm=np.round(dd, 3).tolist(), median_mm=round(float(np.median(dd)), 3),
                            max_mm=round(float(dd.max()), 3))
    allm = dict(metadata=dict(base_md, methods={m: METHODS[m][3] for m in METHODS}, primary=PRIMARY,
                              commands={m: (('UNCRUSH_N=1 ' if METHODS[m][2] else '') + f'{METHODS[m][0]} {METHODS[m][1]} SEED files...') for m in METHODS}),
                skipped_slices=[dict(z=s['z'], n_segments=s['n_segments']) for s in raw['slices'] if not s['used']],
                per_slice=per, smoothed=smoothed, kit_route_agreement=agree)
    (out_dir / 'umbilicus_all_methods.json').write_text(json.dumps(allm))
    qa(name, raw, per, smoothed, kit, out_dir / 'qa.png')
    prim = [r[PRIMARY] for r in per if r[PRIMARY] is not None]
    summ = dict(unit=name, scroll=U['scroll'], volume=U['vol'], vox_um=U['vox_um'], grid_level=0 if f == 1 else 2, f=f,
                shape_zyx=U['shape_zyx'], z_range=[used[0]['z'], used[-1]['z']] if used else None, n_slices_total=len(raw['slices']),
                n_used=len(used), n_skipped=len(raw['slices']) - len(used),
                unsure={m: len(smoothed[m]['unsure_z']) for m in METHODS}, rejected={m: smoothed[m]['n_rejected'] for m in METHODS},
                median_spread_mm={m: round(float(np.median([r[m]['spread_mm'] for r in per if r[m]])), 3) for m in METHODS},
                n_points=len(sm['z']), cp_z_range=[sm['z'][0], sm['z'][-1]] if sm['z'] else None,
                n_trimmed={m: len(smoothed[m]['trimmed_z']) for m in METHODS},
                unsure_in_range={m: smoothed[m]['unsure_in_range'] for m in METHODS},
                rejected_in_range={m: smoothed[m]['rejected_in_range'] for m in METHODS}, kit=None if agree is None else {m: agree[m].get('median_mm') for m in METHODS},
                kit_n=None if agree is None else agree[PRIMARY].get('n'), kit_frame_ok=None if agree is None else agree['frame_matches_volume'],
                kit_band=None if agree is None else agree['kit_z_band'],
                kit_max=None if agree is None else agree[PRIMARY].get('max_mm'),
                method_gap_mm={f'{a}-{b}': round(float(np.median([np.hypot(r[a]['x'] - r[b]['x'], r[a]['y'] - r[b]['y']) * U['vox_um'] / 1000
                                                                for r in per if r[a] and r[b]])), 3)
                               for a, b in (('normal', 'unchanged'), ('normal', 'midpoint'), ('midpoint', 'unchanged'))},
                png_kb=round((out_dir / 'qa.png').stat().st_size / 1024, 1))
    print(json.dumps(summ), flush=True)
    return summ


def qa(name, raw, per, smoothed, kit, png):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import io
    from PIL import Image
    U = raw['U']; f = U['f']; qd = QAG / ukey(name)
    byz = {r['z_grid']: r for r in per}; sm = smoothed[PRIMARY]
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.6))
    colors = {'unchanged': '#888888', 'midpoint': '#2ca02c', 'normal': '#1f77b4'}
    for k, zg in enumerate(raw['qa_z_grid']):
        a = ax.flat[k]; P = [p * f for p in paths((qd / f'{zg:06d}.grid').read_bytes())]
        segs = [p for p in P if len(p) >= 2]
        a.add_collection(LineCollection(segs, colors='k', linewidths=0.25, alpha=0.6))
        allp = np.vstack(segs); lo, hi = np.percentile(allp, 0.5, 0), np.percentile(allp, 99.5, 0); pad = 0.05 * (hi - lo)
        a.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); a.set_ylim(hi[1] + pad[1], lo[1] - pad[1]); a.set_aspect('equal')
        r = byz[zg]; z = r['z']
        for m in METHODS:
            if r[m]: a.plot(r[m]['x'], r[m]['y'], 'x' if m != PRIMARY else '+', color=colors[m], ms=7 if m != PRIMARY else 12, mew=1.5)
        if z in sm['z']:
            i = sm['z'].index(z); a.plot(sm['x'][i], sm['y'][i], 'o', mfc='none', mec='r', ms=14, mew=2)
        if kit is not None and kit['z_band'][0] <= z <= kit['z_band'][1]:
            C = kit['C']; a.plot(np.interp(z, C[:, 0], C[:, 1]), np.interp(z, C[:, 0], C[:, 2]), 'D', mfc='none', mec='orange', ms=10, mew=1.5)
        pr = r[PRIMARY]
        a.set_title(f"z={z}  segs={r['n_segments']}  spread={pr['spread_mm']:.2f} mm{'  UNSURE' if pr['unsure'] else ''}", fontsize=9,
                    color='r' if pr['unsure'] else 'k')
        a.tick_params(labelsize=7)
    zall = np.array([r['z'] for r in per])
    for j, (c, lab) in enumerate((('x', 'x'), ('y', 'y'))):
        a = ax.flat[6 + j]
        for m in METHODS:
            v = np.array([r[m][c] if r[m] else np.nan for r in per])
            a.plot(zall, v, '.', color=colors[m], ms=3 if m != PRIMARY else 5, alpha=0.5 if m != PRIMARY else 0.9,
                   label=f'raw {m}' + (' (primary)' if m == PRIMARY else ''))
        uz = [r['z'] for r in per if r[PRIMARY] and r[PRIMARY]['unsure']]
        a.plot(uz, [r[PRIMARY][c] for r in per if r[PRIMARY] and r[PRIMARY]['unsure']], 'o', mfc='none', mec='r', ms=7, label='UNSURE (primary)')
        rz = sm['rejected_z']; a.plot(rz, [r[PRIMARY][c] for r in per if r['z'] in rz], 'kx', ms=7, label='rejected (> 3 mm)')
        for zt in sm['trimmed_z']: a.axvspan(zt - 100, zt + 100, color='0.85', zorder=0, lw=0)
        a.plot([], [], 's', color='0.85', label='trimmed scan end')
        a.plot(sm['z'], sm[c], 'r-', lw=1.5, label='smoothed (umbilicus.json)')
        if kit is not None:
            C = kit['C']; a.plot(C[:, 0], C[:, 1 + j], '-', color='orange', lw=3, alpha=0.8, label='kit route (skeleton+RANSAC)')
        a.set_xlabel('z (full-res voxels)', fontsize=8); a.set_ylabel(f'{lab} (full-res voxels)', fontsize=8); a.tick_params(labelsize=7)
        if j == 0: a.legend(fontsize=6, loc='best')
    fig.suptitle(f"{name}  {U['vol']}  (grid level {0 if f == 1 else 2}; primary = {METHODS[PRIMARY][3]}; red ring = smoothed, "
                 f"+ = raw primary, x = other methods, orange = kit route)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for dpi in (75, 65, 55, 45):  # keep each PNG <= 300 KB: plain PNG first, then a 256-colour palette
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=dpi); b2 = buf
        if buf.tell() > 300 * 1024:
            im = Image.open(io.BytesIO(buf.getvalue())).convert('RGB'); b2 = io.BytesIO()
            im.quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(b2, format='PNG', optimize=True)
        if b2.tell() <= 300 * 1024: break
    png.write_bytes(b2.getvalue()); plt.close(fig)


if __name__ == '__main__':
    UN = units(); cmd = sys.argv[1]; sel = sys.argv[2:] or list(UN)
    if cmd == 'units':
        for k, v in UN.items(): print(k, json.dumps(v))
    elif cmd == 'run':
        SHM.mkdir(parents=True, exist_ok=True)
        for n in sel:
            if (WORK / f'{ukey(n)}.raw.json').exists() and not os.environ.get('UMB_FORCE'):
                print(n, 'raw exists, skip'); continue
            run_unit(n, UN[n])
    elif cmd == 'build':
        S = [build_unit(n) for n in sel if (WORK / f'{ukey(n)}.raw.json').exists()]
        old = json.loads((W / 'summary.json').read_text()) if (W / 'summary.json').exists() else {}
        old.update({s['unit']: s for s in S}); (W / 'summary.json').write_text(json.dumps(old, indent=1))
