"""Build lasagna-format fit inputs (nx, ny, grad_mag; OME level 2 = 4x; uint8; 32^3 chunks) for a z band of a
scroll from the published m7 surface prediction (m7normals.py: structure-tensor normals + ray-crossing density).
Validated on PHerc1447 against hand-verified organiser sheets (as close to the truth as the published lasagna) and on
PHerc0343 against the published lasagna. CPU only. Label-free.

Output layout = the published lasagna band subsets the fit already reads (spiral-scroll.json: normal_zarr_group "2",
lasagna_scale 4):  OUT/lasagna_inputs/<SCROLL>_{nx,ny,grad_mag}.ome.zarr/2  (+ .zattrs/.zgroup, BAND_SUBSET.json).
Only the band's chunk rows exist (missing chunks = fill 0 = "no data" for the fit).

The m7 level-1 chunk cache lives under $QUIET_WORK/tmp (default $VC_WORK/quiet/tmp) and is deleted row by row as the band advances (peak ~0.5 GB).

usage: build_normals_band.py SCROLL Z0 Z1 OUT_DIR [--sigma-d 1.0] [--sigma-i 2.5] [--coh-min 0.6] [--gm-scale K]
                             [--workers 4] [--no-density] [--cname zstd --clevel 5]
"""
import os, sys, json, time, argparse, shutil, glob
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import zarr, numcodecs
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7normals as M

TMP = M.TMP
H2 = 12
CH = 32
CFG = {}
_m7 = None


def m7():
    global _m7
    if _m7 is None:
        _m7 = M.ChunkMirror(M.M7_TMPL.format(s=CFG['scroll'], v=M.VOL[CFG['scroll']]) + '/1', CFG['cache'], workers=8)
    return _m7


def init(cfg):
    CFG.update(cfg)


def work(job):
    z0, z1, y0, y1, x0, x1 = job
    box = m7().read_box(2 * (z0 - H2), 2 * (z1 + H2), 2 * (y0 - H2), 2 * (y1 + H2), 2 * (x0 - H2), 2 * (x1 + H2))
    if not box.any():
        return job, 0, 0.0
    t = time.time()
    (vz, vy, vx), valid, coh, dens = M.compute_block(box, sigma_d=CFG['sd'], sigma_i=CFG['si'],
                                                      want_density=CFG['density'])
    c = (slice(H2, -H2),) * 3
    val = valid[c] & (coh[c] >= CFG['coh_min'])
    nx8, ny8 = M.encode_u8(vz[c], vy[c], vx[c], val)
    out = {}
    out['nx'], out['ny'] = nx8, ny8
    if CFG['density']:
        gm = np.clip(np.round(dens[c] * CFG['gm_scale']), 0, 255).astype(np.uint8)
        gm[~val] = 0
        out['grad_mag'] = gm
    for name, arr in out.items():
        za = zarr.open(CFG['paths'][name] + '/2', mode='r+')
        za[z0:z1, y0:y1, x0:x1] = arr
    return job, int(val.sum()), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll'); ap.add_argument('z0', type=int); ap.add_argument('z1', type=int); ap.add_argument('out')
    ap.add_argument('--sigma-d', type=float, default=1.0); ap.add_argument('--sigma-i', type=float, default=2.5)
    ap.add_argument('--coh-min', type=float, default=0.6); ap.add_argument('--gm-scale', type=float, default=None)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--no-density', action='store_true')
    ap.add_argument('--tile', type=int, default=256); ap.add_argument('--cname', default='zstd'); ap.add_argument('--clevel', type=int, default=5)
    a = ap.parse_args()
    try:
        os.nice(19 - os.nice(0))
    except Exception:
        pass
    s = a.scroll; v = M.VOL[s]
    density = not a.no_density
    if density and a.gm_scale is None:
        sys.exit('--gm-scale is required with density (calibrated on PHerc0343)')
    las = os.path.join(a.out, 'lasagna_inputs'); os.makedirs(las, exist_ok=True)
    m7l1 = M.ChunkMirror(M.M7_TMPL.format(s=s, v=v) + '/1', f'{TMP}/m7cache_{s}/1')
    Z1l, Y1l, X1l = m7l1.shape
    shp2 = ((Z1l + 1) // 2, (Y1l + 1) // 2, (X1l + 1) // 2)
    comp = numcodecs.Blosc(cname=a.cname, clevel=a.clevel, shuffle=numcodecs.Blosc.SHUFFLE)
    names = ['nx', 'ny'] + (['grad_mag'] if density else [])
    paths = {}
    for n in names:
        p = os.path.join(las, f'{s}_{n}.ome.zarr'); paths[n] = p
        g = zarr.open_group(p, mode='a', zarr_format=2)
        g.attrs['multiscales'] = [{'version': '0.4', 'name': n,
                                   'axes': [{'name': ax, 'type': 'space', 'unit': 'pixel'} for ax in 'zyx'],
                                   'datasets': [{'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4.0, 4.0, 4.0]}]}]}]
        g.attrs['provenance'] = dict(source='published m7 surface prediction (label-free)', method='m7normals.py',
                                                sigma_d=a.sigma_d, sigma_i=a.sigma_i, coh_min=a.coh_min, gm_scale=a.gm_scale)
        if '2' not in g:
            g.create_array('2', shape=shp2, chunks=(CH, CH, CH), dtype='u1', fill_value=0, compressors=comp,
                           chunk_key_encoding={'name': 'v2', 'separator': '/'})
    zb0 = (a.z0 // 4) // CH * CH; zb1 = -(-((a.z1 - 1) // 4 + 1) // CH) * CH
    zb1 = min(zb1, shp2[0])
    json.dump(dict(band_subset=dict(z_full=[a.z0, a.z1], level='2', l2_z=[zb0, zb1], src='m7 structure tensor',
                                    note='band built ' + time.strftime('%Y-%m-%d'))),
              open(os.path.join(las, 'BAND_SUBSET.json'), 'w'), indent=1)
    T = a.tile
    cfg = dict(scroll=s, cache=f'{TMP}/m7cache_{s}/1', sd=a.sigma_d, si=a.sigma_i, coh_min=a.coh_min,
               gm_scale=a.gm_scale, density=density, paths=paths)
    t0 = time.time(); nvalid = 0; cpu = 0.0; nblocks = 0
    with Pool(a.workers, initializer=init, initargs=(cfg,)) as pool:
        for zz in range(zb0, zb1, CH):
            jobs = [(zz, min(zz + CH, shp2[0]), y, min(y + T, shp2[1]), x, min(x + T, shp2[2]))
                    for y in range(0, shp2[1], T) for x in range(0, shp2[2], T)]
            for job, nv, secs in pool.imap_unordered(work, jobs):
                nvalid += nv; cpu += secs; nblocks += 1
            # free level-1 cache rows no longer needed by the next z block (with halo)
            need_lo = 2 * (zz + CH - H2)
            for d in glob.glob(f'{TMP}/m7cache_{s}/1/*'):
                b = os.path.basename(d)
                if b.isdigit() and (int(b) + 1) * 192 <= need_lo:
                    shutil.rmtree(d, ignore_errors=True)
            print(f'z2 {zz}-{zz + CH} done; valid {nvalid}; wall {time.time() - t0:.0f}s cpu {cpu:.0f}s', flush=True)
    shutil.rmtree(f'{TMP}/m7cache_{s}', ignore_errors=True)
    du = {n: sum(os.path.getsize(f) for f in glob.glob(paths[n] + '/2/**', recursive=True) if os.path.isfile(f)) for n in names}
    summ = dict(scroll=s, band=[a.z0, a.z1], l2_z=[zb0, zb1], sigma_d=a.sigma_d, sigma_i=a.sigma_i, coh_min=a.coh_min,
                gm_scale=a.gm_scale, workers=a.workers, wall_s=round(time.time() - t0), block_cpu_s=round(cpu),
                valid_l2_voxels=nvalid, bytes=du, total_GB=round(sum(du.values()) / 1e9, 3), codec=f'blosc-{a.cname}-{a.clevel}')
    json.dump(summ, open(os.path.join(las, 'BUILD_SUMMARY.json'), 'w'), indent=1)
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
