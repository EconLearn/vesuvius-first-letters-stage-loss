"""Fetch a z band of the PUBLISHED lasagna nx/ny/grad_mag (level 2) for PHerc0343 into DST/lasagna_inputs (anonymous
public S3, read-only). The same fetch as tools/g12_fetch_lasagna.py (PHerc1447), with the scroll fixed to PHerc0343 and
the destination as an argument. Disk guard: refuses to start below $VC_MIN_FREE_GB (default 20) GB free.
usage: fetch_lasagna_band.py Z0 Z1 DST_DATASET_DIR"""
import json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import s3fs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))    # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

Z0, Z1, DS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SRC = 'vesuvius-challenge-open-data/PHerc0343/representations/predictions/lasagna/20250521140437-lasagna-20260419180421'
DST = os.path.join(DS, 'lasagna_inputs')
GROUP, SCALE, CH = '2', 4, 32
free = shutil.disk_usage(_P.disk_guard_path()).free / 1e9
if free < _P.MIN_FREE_GB:
    sys.exit(f'disk guard: {_P.disk_guard_path()} free {free:.0f} GB < {_P.MIN_FREE_GB:.0f}')
fs = s3fs.S3FileSystem(anon=True, default_block_size=1 << 20)
r0, r1 = (Z0 // SCALE) // CH, ((Z1 - 1) // SCALE) // CH
os.makedirs(DST, exist_ok=True)
json.dump({'band_subset': dict(z_full=[Z0, Z1], level=GROUP, chunk_rows=[r0, r1], src=SRC)},
          open(os.path.join(DST, 'BAND_SUBSET.json'), 'w'), indent=1)
for name in ['nx', 'ny', 'grad_mag']:
    store = f'PHerc0343_{name}.ome.zarr'
    src, dst = f'{SRC}/{store}', os.path.join(DST, store)
    os.makedirs(os.path.join(dst, GROUP), exist_ok=True)
    for f in ['.zattrs', '.zgroup']:
        open(os.path.join(dst, f), 'wb').write(fs.cat_file(f'{src}/{f}'))
    open(os.path.join(dst, GROUP, '.zarray'), 'wb').write(fs.cat_file(f'{src}/{GROUP}/.zarray'))
    keys = []
    for r in range(r0, r1 + 1):
        keys += fs.find(f'{src}/{GROUP}/{r}/')

    def get(k):
        out = os.path.join(dst, k[len(src) + 1:])
        if os.path.exists(out):
            return 0
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for a in range(6):
            try:
                b = fs.cat_file(k); break
            except Exception:
                time.sleep(2 * (a + 1))
        else:
            raise RuntimeError('failed ' + k)
        open(out + '.part', 'wb').write(b); os.replace(out + '.part', out)
        return len(b)
    t = time.time()
    with ThreadPoolExecutor(48) as ex:
        tot = sum(ex.map(get, keys))
    print(name, len(keys), f'{tot / 1e6:.0f} MB', round(time.time() - t), 's', flush=True)
print('ALL DONE')
