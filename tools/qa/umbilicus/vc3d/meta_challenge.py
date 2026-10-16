import s3fs, json, sys
fs = s3fs.S3FileSystem(anon=True); L = json.load(open(sys.argv[1] + '/listing.json')); D = sys.argv[1]
for s, v in L.items():
    u = json.loads(fs.cat_file(v['umb'][0])); open(f'{D}/{s}_umbilicus.json', 'w').write(json.dumps(u))
    cp = u.get('control_points', u if isinstance(u, list) else [])
    print('==', s, 'umb keys', [k for k in u if k != 'control_points'] if isinstance(u, dict) else 'list', 'n', len(cp), 'first', cp[:1], 'last', cp[-1:])
    if isinstance(u, dict): print('   meta', {k: u[k] for k in u if k != 'control_points'})
    ng = v['ng'][0]
    try: print('   ng ls', [p.split('/')[-1] for p in fs.ls(ng)][:10])
    except Exception as e: print('   ng ls fail', e)
    for f in ('metadata.json', 'meta.json'):
        try: print('   ng', f, fs.cat_file(f'{ng}/{f}')[:600].decode()); break
        except Exception: pass
    try:
        xs = sorted(p.split('/')[-1] for p in fs.ls(f'{ng}/xy')); print('   xy n', len(xs), xs[:2], xs[-2:])
    except Exception as e: print('   xy fail', e)
