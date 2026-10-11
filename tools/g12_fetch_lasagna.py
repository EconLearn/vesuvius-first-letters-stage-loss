"""G1: fetch the PHerc1447 lasagna normal/grad_mag band (level 2 = 4x down) for the fit window.

Only chunk rows covering full-res z [Z0, Z1) are fetched (anonymous public S3, read-only).
Writes a valid OME-Zarr subset store per channel under the fit dataset's lasagna_inputs/.
Disk guard: refuses to start if the data disk has < $VC_MIN_FREE_GB (default 20) GB free.
usage: g12_fetch_lasagna.py Z0 Z1
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import s3fs

Z0, Z1 = int(sys.argv[1]), int(sys.argv[2])
SRC = 'vesuvius-challenge-open-data/PHerc1447/representations/predictions/lasagna/20250521151220-lasagna-20260419180421'
DST = f'{_P.DATA}/spiral/PHerc1447/lasagna_inputs'
GROUP, SCALE, CH = '2', 4, 32
free = shutil.disk_usage(_P.disk_guard_path()).free / 1e9
if free < _P.MIN_FREE_GB:
    sys.exit(f'disk guard: {_P.disk_guard_path()} free {free:.0f} GB < {_P.MIN_FREE_GB:.0f}')
fs = s3fs.S3FileSystem(anon=True, default_block_size=1 << 20)
r0, r1 = (Z0 // SCALE) // CH, ((Z1 - 1) // SCALE) // CH
print('chunk rows', r0, r1, flush=True)
os.makedirs(DST, exist_ok=True)
json.dump({'band_subset': dict(z_full=[Z0, Z1], level=GROUP, chunk_rows=[r0, r1], src=SRC,
                               note='G1/G2 band-only subset, fetched 2026-10-07')},
          open(os.path.join(DST, 'BAND_SUBSET.json'), 'w'), indent=1)
for name in ['nx', 'ny', 'grad_mag']:
    store = f'PHerc1447_{name}.ome.zarr'
    src, dst = f'{SRC}/{store}', os.path.join(DST, store)
    os.makedirs(os.path.join(dst, GROUP), exist_ok=True)
    for f in ['.zattrs', '.zgroup']:
        open(os.path.join(dst, f), 'wb').write(fs.cat_file(f'{src}/{f}'))
    open(os.path.join(dst, GROUP, '.zarray'), 'wb').write(fs.cat_file(f'{src}/{GROUP}/.zarray'))
    t = time.time(); keys = []
    for r in range(r0, r1 + 1):
        keys += [k for k in fs.find(f'{src}/{GROUP}/{r}/')]
    print(name, 'keys', len(keys), 'listed in', round(time.time() - t, 1), 's', flush=True)

    def get(k):
        rel = k[len(src) + 1:]
        out = os.path.join(dst, rel)
        if os.path.exists(out):
            return 0
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for a in range(6):
            try:
                b = fs.cat_file(k)
                break
            except Exception:
                time.sleep(2 * (a + 1))
        else:
            raise RuntimeError('failed ' + k)
        tmp = out + '.part'
        open(tmp, 'wb').write(b); os.replace(tmp, out)
        return len(b)

    t = time.time(); tot = 0
    with ThreadPoolExecutor(48) as ex:
        for i, n in enumerate(ex.map(get, keys)):
            tot += n
            if i % 5000 == 0:
                print(f'  {name} {i}/{len(keys)} {tot/1e6:.0f} MB {time.time()-t:.0f}s', flush=True)
    print(name, 'DONE', len(keys), f'{tot/1e6:.0f} MB', round(time.time() - t), 's', flush=True)
print('ALL DONE', flush=True)
