"""Package the organiser's PHerc1447 w058 / w060 surface layers (HF ink-8um-pherc1447-surfaces, 24 uint8 tifs) as render
zarrs (group, array '0' (24, H, W), chunks (24, 128, 128)) for depth_sweep_fast; d = k - 12.
usage: w1447_prep.py HF_SURFACES_DIR OUT_DIR"""
import glob, sys, json, numpy as np, tifffile, zarr
H, O = sys.argv[1], sys.argv[2]
for w in ('w058', 'w060'):
    fs = sorted(glob.glob(f'{H}/{w}/layers/*.tif')); a = np.stack([tifffile.imread(f) for f in fs])
    g = zarr.open_group(f'{O}/r_{w}.zarr', mode='w', zarr_format=2)
    arr = g.create_array('0', shape=a.shape, chunks=(len(fs), 128, 128), dtype='uint8'); arr[:] = a
    g.attrs['ensemble_render'] = dict(n_layers=len(fs), step=1.0, d0_layer=len(fs) // 2, depth_convention=f'd = k - {len(fs) // 2}',
                                      source=f'HF ink-8um-pherc1447-surfaces/{w}/layers (organiser surface volume)')
    print(w, a.shape, a.dtype)
