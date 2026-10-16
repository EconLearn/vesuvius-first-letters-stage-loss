"""Per normal grid: xy file sizes from the S3 listing (no downloads) -> umbkit/xy_sizes.json; plus volume metadata.json."""
import json, time, s3fs
fs = s3fs.S3FileSystem(anon=True); B = 'vesuvius-challenge-open-data'; W = str(__import__('pathlib').Path(__file__).resolve().parents[1])
inv = json.load(open(f'{W}/s3_inventory.json')); out = {}; vm = {}
def retry(f, *a, **k):
    for att in range(5):
        try: return f(*a, **k)
        except Exception as e: err = e; time.sleep(3 * (att + 1))
    raise err
for s, r in inv.items():
    for n in r['m7_normal_grids']:
        ent = retry(fs.ls, f'{B}/{s}/representations/predictions/surfaces/{n}/xy', detail=True)
        d = {int(e['name'].split('/')[-1].split('.')[0]): e['size'] for e in ent if e['name'].endswith('.grid')}
        out[f'{s}|{n}'] = d
        sz = sorted(d.values()); nz = [k for k, v in d.items() if v > 200]
        print(s, n[:14], 'n', len(d), 'min', sz[0], 'median', sz[len(sz)//2], 'max', sz[-1], 'n>200B', len(nz), 'range', min(nz) if nz else None, max(nz) if nz else None, flush=True)
    for v in r['zarr']:
        try: vm[f'{s}|{v}'] = json.loads(retry(fs.cat_file, f'{B}/{s}/volumes/{v}/metadata.json'))
        except Exception as e: vm[f'{s}|{v}'] = {'err': str(e)}
json.dump(out, open(f'{W}/xy_sizes.json', 'w')); json.dump(vm, open(f'{W}/volume_metadata.json', 'w'), indent=1)
print(json.dumps(list(vm.values())[0], indent=1)[:2500])
