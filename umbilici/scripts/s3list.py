"""List S3 prefixes, normal grids and volume metadata for the 9 target scrolls -> umbkit/s3_inventory.json"""
import json, time, s3fs
fs = s3fs.S3FileSystem(anon=True); B = 'vesuvius-challenge-open-data'
OUT = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 's3_inventory.json')
T = ['PHerc0175A', 'PHerc0175B', 'PHerc0306B', 'PHerc0483A', 'PHerc0483B', 'PHerc0490A', 'PHerc0490B', 'PHerc0846A', 'PHerc0846B']
def retry(f, *a, **k):
    for att in range(5):
        try: return f(*a, **k)
        except FileNotFoundError: raise
        except Exception as e:
            err = e; time.sleep(3 * (att + 1))
    raise err
tops = set(p.split('/')[-1] for p in retry(fs.ls, B))
inv = {}
for s in T:
    r = {'top_level': s in tops}
    print('==', s, 'top-level', r['top_level'], flush=True)
    if not r['top_level']:
        inv[s] = r; continue
    try: r['children'] = [p.split('/')[-1] for p in retry(fs.ls, f'{B}/{s}')]
    except Exception as e: r['children'] = str(e)
    try: r['volumes'] = [p.split('/')[-1] for p in retry(fs.ls, f'{B}/{s}/volumes')]
    except Exception as e: r['volumes'] = str(e)
    try: r['umbilicus_dir'] = [p.split('/')[-1] for p in retry(fs.ls, f'{B}/{s}/representations/umbilicus')]
    except Exception as e: r['umbilicus_dir'] = 'none: ' + type(e).__name__
    try: surf = [p.split('/')[-1] for p in retry(fs.ls, f'{B}/{s}/representations/predictions/surfaces')]
    except Exception as e: surf = []; r['surf_err'] = str(e)
    r['surfaces'] = surf
    ng = [n for n in surf if n.endswith('.normal-grids') and '-surface-m7-' in n]
    r['m7_normal_grids'] = ng
    r['ng'] = {}
    for n in ng:
        d = {}
        base = f'{B}/{s}/representations/predictions/surfaces/{n}'
        try: d['root'] = [p.split('/')[-1] for p in retry(fs.ls, base)]
        except Exception as e: d['root'] = str(e)
        for f in ('metadata.json', 'meta.json'):
            try: d[f] = retry(fs.cat_file, f'{base}/{f}')[:2000].decode(); break
            except Exception: pass
        try:
            xs = sorted(p.split('/')[-1] for p in retry(fs.ls, f'{base}/xy'))
            g = [x for x in xs if x.endswith('.grid')]
            d['xy_n'] = len(g); d['xy_first'] = g[:3]; d['xy_last'] = g[-3:]
            d['xy_nums'] = [int(x.split('.')[0]) for x in g]
        except Exception as e: d['xy_err'] = str(e)
        r['ng'][n] = d
    r['zarr'] = {}
    for v in (r['volumes'] if isinstance(r['volumes'], list) else []):
        try:
            vb = f'{B}/{s}/volumes/{v}'
            z = {'root': [p.split('/')[-1] for p in retry(fs.ls, vb)][:20]}
            for f in ('.zattrs', 'zarr.json', '.zgroup', 'meta.json'):
                try: z[f] = retry(fs.cat_file, f'{vb}/{f}')[:3000].decode()
                except Exception: pass
            try: z['0/.zarray'] = retry(fs.cat_file, f'{vb}/0/.zarray').decode()
            except Exception: pass
            r['zarr'][v] = z
        except Exception as e: r['zarr'][v] = str(e)
    inv[s] = r
    print('  volumes', r['volumes'], '| m7 ng', ng, '| xy_n', {k: v.get('xy_n') for k, v in r['ng'].items()}, '| umb', r['umbilicus_dir'], flush=True)
json.dump(inv, open(OUT, 'w'), indent=1)
print('wrote', OUT)
