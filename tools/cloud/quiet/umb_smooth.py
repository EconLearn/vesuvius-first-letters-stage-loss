"""Smooth a band's automatic umbilicus (Huber straight line in z through all per-seed estimates), write the fit dataset
files (umbilicus.json, spiral-scroll.json) and a QA image (CT level-3 slice at the band centre + per-seed estimates
(yellow) + the line (magenta) + m7 L3 sheets (faint red)).

spiral-scroll.json: catalogue z-direction / handedness where the catalogue has them (column z_top_to_bottom of the
screening config, m7normals.ZDIR). Where the catalogue has none, spiral-scroll.json is written WITHOUT a sense (the fit
refuses to start) plus two ready variants spiral-scroll.CW.json / spiral-scroll.ACW.json; a person picks one from the
core-zoom QA image (an automatic sense estimate did not pass validation). jobs/job3_quiet.sh can instead fit both variants and pick the
better one (SENSE_MODE, quiet/pick_sense.py).

usage: umb_smooth.py SCROLL RAW_JSON DATASET_DIR Z0 Z1 [--sense-json PATH] [--qa PNG]
"""
import os, sys, json, argparse
import numpy as np
import s3fs, zarr
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7normals as M

SHAPE, ZDIR = M.SHAPE, M.ZDIR          # built-in public scrolls + the screening config (see m7normals.py)
CT = 'vesuvius-challenge-open-data/{s}/volumes/{vid}'      # vid = the config's vol_id (any voxel size)


def huber_line(z, v, zc, it=50, k=100.0):
    w = np.ones_like(v)
    for _ in range(it):
        A = np.stack([np.ones_like(z), z - zc], 1) * np.sqrt(w)[:, None]
        c = np.linalg.lstsq(A, v * np.sqrt(w), rcond=None)[0]
        r = v - (c[0] + c[1] * (z - zc))
        w = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-9))
    return c, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll'); ap.add_argument('raw'); ap.add_argument('ds'); ap.add_argument('z0', type=int); ap.add_argument('z1', type=int)
    ap.add_argument('--sense-json', default=None); ap.add_argument('--qa', default=None)
    a = ap.parse_args()
    s = a.scroll; v = M.VOL[s]; vid = M.VOLID[s]; vox = M.VOXEL[s]
    os.makedirs(a.ds, exist_ok=True)
    U = json.load(open(a.raw))
    zc = float((a.z0 + a.z1) // 2)
    Z, Y, X = [], [], []
    for r in U['provenance']['per_slice']:
        for y, x in r['yx_per_seed']:
            Z.append(r['z']); Y.append(y); X.append(x)
    Z, Y, X = map(np.asarray, (Z, Y, X))
    cy, ry = huber_line(Z, Y, zc); cx, rx = huber_line(Z, X, zc)
    zs = list(range(a.z0 - 200, a.z1 + 201, 100))
    cps = [dict(x=round(float(cx[0] + cx[1] * (z - zc)), 1), y=round(float(cy[0] + cy[1] * (z - zc)), 1), z=z, score=100) for z in zs]
    prov = dict(U['provenance'], smoothing='Huber (k=100 vox) line in z, all per-seed estimates',
                line_y=[float(cy[0]), float(cy[1])], line_x=[float(cx[0]), float(cx[1])], zc=zc,
                resid_mad_y=float(np.median(np.abs(ry))), resid_mad_x=float(np.median(np.abs(rx))),
                seed_spread_vox_max=float(max(r['spread_vox'] for r in U['provenance']['per_slice'])))
    json.dump(dict(control_points=cps, provenance=prov), open(os.path.join(a.ds, 'umbilicus.json'), 'w'), indent=1)
    spec = {"schema_version": 1, "name": s, "voxel_size_um": vox,
            "z_direction_is_top_to_bottom": ZDIR[s], "left_handed_coordinates": False,
            "base_shape_zyx": SHAPE[s], "normal_zarr_group": "2", "lasagna_scale": 4,
            "paths": {"normal_x": f"lasagna_inputs/{s}_nx.ome.zarr", "normal_y": f"lasagna_inputs/{s}_ny.ome.zarr",
                      "gradient_magnitude": f"lasagna_inputs/{s}_grad_mag.ome.zarr"}}
    if ZDIR[s] is None:
        spec['_sense_note'] = ('NO SENSE: the catalogue has no z-direction for this scroll; copy spiral-scroll.CW.json or '
                               'spiral-scroll.ACW.json over this file after a person reads the core-zoom QA image, or let job 3 fit both '
                               'senses and pick one (SENSE_MODE)')
        for sense in ('CW', 'ACW'):
            var = dict(spec); var.pop('_sense_note'); var['spiral_outward_sense'] = sense
            json.dump(var, open(os.path.join(a.ds, f'spiral-scroll.{sense}.json'), 'w'), indent=1)
    json.dump(spec, open(os.path.join(a.ds, 'spiral-scroll.json'), 'w'), indent=1)
    print('y(z) = %.1f + %.4f (z-%d); MAD %.0f | x(z) = %.1f + %.4f (z-%d); MAD %.0f' % (cy[0], cy[1], zc, prov['resid_mad_y'], cx[0], cx[1], zc, prov['resid_mad_x']))
    # QA image
    fs = s3fs.S3FileSystem(anon=True)
    ct = zarr.open(fs.get_mapper(CT.format(s=s, vid=vid) + '/3'), mode='r')
    m3 = zarr.open(fs.get_mapper(M.M7_TMPL.format(s=s, v=v) + '/3'), mode='r')
    zq = int(zc)
    sl = np.asarray(ct[zq // 8]); ms = np.asarray(m3[zq // 8]) > 0
    rgb = np.stack([sl, sl, sl], -1).astype(np.float32)
    rgb[ms, 0] = 0.6 * rgb[ms, 0] + 0.4 * 255
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(img)
    for r in U['provenance']['per_slice']:
        for y, x in r['yx_per_seed']:
            dr.ellipse([x / 8 - 2, y / 8 - 2, x / 8 + 2, y / 8 + 2], outline=(255, 255, 0))
    ux, uy = (cx[0] + cx[1] * (zq - zc)) / 8, (cy[0] + cy[1] * (zq - zc)) / 8
    dr.ellipse([ux - 6, uy - 6, ux + 6, uy + 6], outline=(255, 0, 255), width=2)
    qa = a.qa or os.path.join(a.ds, f'umbilicus_qa_z{zq}.png')
    # crop around the scroll for readability
    yy, xx = np.nonzero(sl > 0)
    if len(yy):
        img = img.crop((max(int(xx.min()) - 20, 0), max(int(yy.min()) - 20, 0), int(xx.max()) + 20, int(yy.max()) + 20))
    img.save(qa)
    # core zoom for a human spiral-sense read: CT level 1 (2 x the voxel size), +-700 full-res voxels around the umbilicus
    ct1 = zarr.open(fs.get_mapper(CT.format(s=s, vid=vid) + '/1'), mode='r')
    uyf, uxf = cy[0], cx[0]
    y0, x0 = int(max(uyf - 700, 0)) // 2, int(max(uxf - 700, 0)) // 2
    crop = np.asarray(ct1[zq // 2, y0:y0 + 700, x0:x0 + 700])
    zi = Image.fromarray(crop).convert('RGB'); d2 = ImageDraw.Draw(zi)
    py, px = uyf / 2 - y0, uxf / 2 - x0
    d2.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(255, 0, 255), width=2)
    d2.line([20, 680, 20 + int(round(1000.0 / (2 * vox))), 680], fill=(255, 255, 0), width=3)  # 1 mm at level 1 (58 px at 8.64 um)
    d2.text((20, 660), '1 mm; image x right, y down (array order); z = %d' % zq, fill=(255, 255, 0))
    zq_path = qa.replace('.png', '_core_zoom.png')
    zi.save(zq_path)
    print('wrote', os.path.join(a.ds, 'umbilicus.json'), os.path.join(a.ds, 'spiral-scroll.json'), qa, zq_path)


if __name__ == '__main__':
    main()
