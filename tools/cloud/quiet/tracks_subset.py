"""Stream ONLY the band's tracks out of the organisers' published spiral-fit track store (dl.ash2txt.org
datasets/spiral_datasets/<scroll>/<volume>/tracks/*.dbm.vctracks, 6-10 GiB per scroll) with HTTP range requests.

The packed store is a set of flat arrays (tracks.py write_packed_track_store): offsets.i64 (track_count+1),
z_bounds.i32 (track_count x 2), family_codes.i8, coordinates.i32 (points x 3, zyx). We stream z_bounds / offsets /
family_codes into RAM, keep the tracks the fit would load for the band (fit_spiral -> load_tracks_from_dbm keeps
tracks lying ENTIRELY inside [z_begin, z_end)), and range-read only their coordinates.

Modes:
  --index-only : write <out>/<scroll>_tracks_<z0>_<z1>.index.npz (selected rows, sizes) + .json summary; no coords.
  default      : also write a legacy pickle DBM <out>/tracks/<scroll>_m7_band<z0>_<z1>.dbm (the format fit_spiral and
                 build_track_crossings.py read; keys 'h:<n>' / 'vx:<n>', 1,000 tracks per key, original order kept).
Then build the crossing cache with the official script (spiral-fitting build_track_crossings.py).

usage: tracks_subset.py SCROLL Z0 Z1 OUT_DIR [--index-only] [--workers 8] [--gap-kib 256]
"""
import os, sys, json, time, argparse, dbm, pickle, struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7normals as M

BASE = 'https://dl.ash2txt.org/datasets/spiral_datasets/{s}/{v}/tracks/{s}_{v}_surface_m7_L0_th0.2.dbm.vctracks/'
PIECE = 16 << 20


def get_range(url, a, b, tries=6):
    """bytes [a, b) of url"""
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b - 1}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != b - a:
                raise IOError(f'short read {len(data)} != {b - a}')
            return data
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def get_array(url, nbytes, dtype, workers):
    buf = bytearray(nbytes)
    pieces = [(a, min(a + PIECE, nbytes)) for a in range(0, nbytes, PIECE)]

    def one(p):
        buf[p[0]:p[1]] = get_range(url, p[0], p[1])
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, pieces))
    return np.frombuffer(bytes(buf), dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll'); ap.add_argument('z0', type=int); ap.add_argument('z1', type=int); ap.add_argument('out')
    ap.add_argument('--index-only', action='store_true'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gap-kib', type=int, default=256, help='merge selected byte ranges closer than this')
    a = ap.parse_args()
    s = a.scroll; v = M.VOL[s]; base = BASE.format(s=s, v=v)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    meta = json.loads(urllib.request.urlopen(base + 'metadata.json', timeout=60).read())
    hdr = urllib.request.urlopen(base + 'header.bin', timeout=60).read()
    magic, ver, hlen, ntr, npts = struct.unpack('<8sIIQQ', hdr[:32])
    assert magic == b'VCTRK01\0' and ntr == meta['track_count'] and npts == meta['point_count'], (magic, ntr, npts, meta)
    zb = get_array(base + 'z_bounds.i32', ntr * 8, '<i4', a.workers).reshape(ntr, 2)
    off = get_array(base + 'offsets.i64', (ntr + 1) * 8, '<i8', a.workers)
    fam = get_array(base + 'family_codes.i8', ntr, 'i1', a.workers)
    t_meta = time.time() - t0
    sel = np.flatnonzero((zb[:, 0] >= a.z0) & (zb[:, 1] < a.z1))
    lens = off[sel + 1] - off[sel]
    npts_sel = int(lens.sum())
    summary = dict(scroll=s, volume=v, band=[a.z0, a.z1], source=base, store_tracks=int(ntr), store_points=int(npts),
                   selected_tracks=int(len(sel)), selected_points=int(npts_sel),
                   selected_horizontal=int((fam[sel] == 0).sum()), selected_vertical=int((fam[sel] == 1).sum()),
                   coords_bytes=npts_sel * 12, coords_GiB=round(npts_sel * 12 / 2 ** 30, 3),
                   overlapping_but_not_inside=int(((zb[:, 1] >= a.z0) & (zb[:, 0] < a.z1)).sum() - len(sel)),
                   meta_stream_s=round(t_meta, 1), meta_stream_MB=round((ntr * 17 + 8) / 1e6))
    np.savez_compressed(os.path.join(a.out, f'{s}_tracks_{a.z0}_{a.z1}.index.npz'), rows=sel, lens=lens.astype(np.int32))
    if a.index_only:
        json.dump(summary, open(os.path.join(a.out, f'{s}_tracks_{a.z0}_{a.z1}.index.json'), 'w'), indent=1)
        print(json.dumps(summary)); return
    # merge row ranges into HTTP ranges (selected rows are mostly contiguous runs)
    starts = off[sel] * 12; ends = off[sel + 1] * 12
    runs = []
    cs, ce = int(starts[0]), int(ends[0])
    for st, en in zip(starts[1:], ends[1:]):
        if st - ce <= (a.gap_kib << 10) and en - cs <= (64 << 20):
            ce = int(en)
        else:
            runs.append((cs, ce)); cs, ce = int(st), int(en)
    runs.append((cs, ce))
    fetched = sum(e - b for b, e in runs)
    coords_url = base + 'coordinates.i32'
    data = {}

    def one(r):
        data[r[0]] = get_range(coords_url, r[0], r[1])
    t1 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, runs))
    t_coords = time.time() - t1
    run_starts = np.array([r[0] for r in runs])
    os.makedirs(os.path.join(a.out, 'tracks'), exist_ok=True)
    out_dbm = os.path.join(a.out, 'tracks', f'{s}_m7_band{a.z0}_{a.z1}.dbm')
    groups = {0: [], 1: []}
    n_keys = 0
    with dbm.open(out_dbm, 'n') as db:
        for i, (row, st, en) in enumerate(zip(sel, starts, ends)):
            k = int(np.searchsorted(run_starts, st, side='right') - 1)
            b0 = run_starts[k]
            arr = np.frombuffer(data[b0][st - b0:en - b0], dtype='<i4').reshape(-1, 3).astype(np.int32)
            f = int(fam[row])
            if f not in groups:
                continue
            groups[f].append(arr)
            if len(groups[f]) == 1000:
                db[f"{'h' if f == 0 else 'vx'}:{n_keys:08d}"] = pickle.dumps(groups[f]); n_keys += 1; groups[f] = []
        for f, g in groups.items():
            if g:
                db[f"{'h' if f == 0 else 'vx'}:{n_keys:08d}"] = pickle.dumps(g); n_keys += 1
    du = sum(os.path.getsize(os.path.join(os.path.dirname(out_dbm), x)) for x in os.listdir(os.path.dirname(out_dbm))
             if x.startswith(os.path.basename(out_dbm)))
    summary.update(http_runs=len(runs), gap_kib=a.gap_kib, fetched_bytes=int(fetched), coords_stream_s=round(t_coords, 1), dbm=out_dbm,
                   dbm_keys=n_keys, dbm_bytes=int(du), dbm_backend=dbm.whichdb(out_dbm))
    json.dump(summary, open(os.path.join(a.out, f'{s}_tracks_{a.z0}_{a.z1}.json'), 'w'), indent=1)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
