#!/usr/bin/env python
"""ensemble_maps: his ink/scripts/ensemble_maps.py logic on our depth-sweep outputs.

  * valid pixel = CT > 0 in every rendered layer;
  * map -> value in [0,1]: villa uint8 clip((v - 64) / 128, 0, 1); Hecate v / 255 bilinearly resized to the render
    grid; v8-in v / 65535;
  * ONE contrast stretch per model across all stat depths: p1 / p99.5 of the reading-side valid pixels pooled over
    --stat-depths (per-model display PNGs);
  * per-model z = (value - mean) / (sd + 1e-6), mean/sd pooled over the same pixels and depths (raw values, not the
    stretched ones);
  * ensemble = mean of the members' z (only at depths where every member exists); its stretch = p1 / p99.5 over the
    fixed grid (every 8th row/col from 4, valid) at those depths;
  * CT layer 33 + d stretched by p1 / p99.5 of valid pixels in layers 18..48.
Extra (ours): --combine mean|max adds a depth-aggregated ensemble over --write-depths ('dmean' / 'dmax').

Outputs in --out: <name>__ensemble__d+NN.npy (float32 z-mean) + .png (stretched u8), <name>__<model>__d+NN.png,
<name>__ct__d+NN.png, optional <name>__ensemble__dmax|dmean.npy/.png, <name>__stats.json. Raw render orientation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens import adapters as A, common as C  # noqa: E402

P_LO, P_HI, SUB = 1.0, 99.5, 8


def dtag(d):
    return f'd{int(d):+03d}'


def read_unit(path, model, hw):
    m = tifffile.imread(path)
    v = A.to_unit(model, m)
    if v.shape != tuple(hw):
        import cv2
        v = cv2.resize(v, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return v


def stretch(v, lo, hi):
    return (np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def write_png(path, u8):
    import cv2
    cv2.imwrite(str(path), u8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render', required=True)
    ap.add_argument('--maps', required=True, help='depth_sweep --out directory')
    ap.add_argument('--models', default='ensemble4')
    ap.add_argument('--order', default='reverse', help='reading side (his: reverse)')
    ap.add_argument('--tag', default='ph0_0', help='ph<r>_<c> or merged')
    ap.add_argument('--stat-depths', default='-15..15')
    ap.add_argument('--write-depths', default='0')
    ap.add_argument('--combine', choices=('none', 'mean', 'max'), default='none')
    ap.add_argument('--out', required=True)
    ap.add_argument('--name')
    ap.add_argument('--no-png', action='store_true')
    ap.add_argument('--stat-mask', help='bool (H, W) npy: statistics and outputs only on these pixels (need mask)')
    a = ap.parse_args(argv)
    models = A.ENSEMBLE4 if a.models == 'ensemble4' else a.models.split(',')
    arr, rattrs = C.open_render(a.render)
    n, H, W = (int(v) for v in arr.shape)
    mid = n // 2
    name = a.name or Path(a.render).name.replace('.zarr', '')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rk = rattrs.get('ensemble_render', {}).get('rendered_k', [0, n])
    valid = C.valid_mask_all_layers(arr, k_range=rk)
    if a.stat_mask:
        valid &= np.load(a.stat_mask).astype(bool)
    mp = lambda m, d: Path(a.maps) / 'maps' / m / f'{m}__{a.order}__{a.tag}__{dtag(d)}.tif'  # noqa: E731
    stat_d = C.parse_depths(a.stat_depths)
    write_d = C.parse_depths(a.write_depths)
    stats = dict(name=name, render=str(a.render), models=models, order=a.order, tag=a.tag, valid_px=int(valid.sum()),
                 stat_depths_requested=stat_d, per_model={}, depth_convention='d = layer - 33, +d away from axis')
    for m in models:
        ds = [d for d in stat_d if mp(m, d).exists()]
        if not ds:
            stats['per_model'][m] = dict(missing=True); continue
        vals, s1, s2, cnt = [], 0.0, 0.0, 0
        for d in ds:
            v = read_unit(mp(m, d), m, (H, W))[valid].astype(np.float64)
            s1 += v.sum(); s2 += (v * v).sum(); cnt += v.size
            vals.append(v.astype(np.float32))
        allv = np.concatenate(vals)
        lo, hi = (float(x) for x in np.percentile(allv, [P_LO, P_HI]))
        mu = s1 / cnt
        sd = float(np.sqrt(max(s2 / cnt - mu * mu, 0.0))) + 1e-6
        stats['per_model'][m] = dict(stat_depths=ds, n=int(cnt), p1=lo, p99_5=hi, mean=float(mu), sd=sd)
    have = [m for m in models if not stats['per_model'][m].get('missing')]
    if not have:
        sys.exit('no maps found')
    full_d = [d for d in stat_d if all(mp(m, d).exists() for m in models)]
    stats['ensemble_depths'] = full_d
    sub = np.zeros((H, W), bool); sub[SUB // 2::SUB, SUB // 2::SUB] = True; sub &= valid

    def zmap(m, d):
        s = stats['per_model'][m]
        return (read_unit(mp(m, d), m, (H, W)) - s['mean']) / s['sd']

    def ens(d):
        E = None
        for m in models:
            z = zmap(m, d)
            E = z if E is None else E + z
        return (E / len(models)).astype(np.float32)
    if full_d:
        g = np.concatenate([ens(d)[sub] for d in full_d])
        lo, hi = (float(x) for x in np.percentile(g, [P_LO, P_HI]))
        stats['ensemble'] = dict(p1=lo, p99_5=hi, depths=full_d, grid_px=int(g.size))
    # CT stretch: layers mid-15..mid+15
    kl0, kl1 = max(rk[0], mid - 15), min(rk[1], mid + 16)
    ctv = np.concatenate([np.asarray(arr[k])[valid] for k in range(kl0, kl1)])
    ct_lo, ct_hi = (float(x) for x in np.percentile(ctv, [P_LO, P_HI]))
    stats['ct'] = dict(p1=ct_lo, p99_5=ct_hi, layers=[kl0, kl1 - 1])
    agg = []
    for d in write_d:
        k = mid + d
        if 0 <= k < n and not a.no_png:
            ct = stretch(np.asarray(arr[k]).astype(np.float32), ct_lo, ct_hi); ct[~valid] = 0
            write_png(out / f'{name}__ct__{dtag(d)}.png', ct)
        for m in have:
            if mp(m, d).exists() and not a.no_png:
                s = stats['per_model'][m]
                u = stretch(read_unit(mp(m, d), m, (H, W)), s['p1'], s['p99_5']); u[~valid] = 0
                write_png(out / f'{name}__{m}__{dtag(d)}.png', u)
        if d in full_d:
            E = ens(d)
            np.save(out / f'{name}__ensemble__{dtag(d)}.npy', E)
            agg.append(E)
            if not a.no_png:
                u = stretch(E, stats['ensemble']['p1'], stats['ensemble']['p99_5']); u[~valid] = 0
                write_png(out / f'{name}__ensemble__{dtag(d)}.png', u)
    if a.combine != 'none' and agg:
        Ec = (np.max(agg, 0) if a.combine == 'max' else np.mean(agg, 0)).astype(np.float32)
        tagc = f'd{a.combine}'
        lo, hi = (float(x) for x in np.percentile(Ec[sub], [P_LO, P_HI]))
        stats[f'ensemble_{tagc}'] = dict(p1=lo, p99_5=hi, over_depths=[d for d in write_d if d in full_d])
        np.save(out / f'{name}__ensemble__{tagc}.npy', Ec)
        if not a.no_png:
            u = stretch(Ec, lo, hi); u[~valid] = 0
            write_png(out / f'{name}__ensemble__{tagc}.png', u)
    np.save(out / f'{name}__valid.npy', valid)
    (out / f'{name}__stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != 'per_model'}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
