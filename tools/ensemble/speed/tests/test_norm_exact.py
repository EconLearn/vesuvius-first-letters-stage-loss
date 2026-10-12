#!/usr/bin/env python
"""Exactness of RobustNormTorch vs villa's numpy normalize_flat_patch ('tifxyz_robust') on real 343 render patches.
Samples patches on the villa grid (17-layer windows, d in -6..+6, reading order), half of them on the edge of the
rendered area (zero padding, the fallback-prone case), plus synthetic edge cases. Reports bit-identical fraction and
max |diff|. A direct percentile test (2026-10-12) compares the clip bounds with np.percentile where the 1st/99th
percentile falls between two different values (a float64 lerp fails it: 1 ulp off numpy 2.5.3 for 14 % / 7.7 % of
value pairs). usage: test_norm_exact.py [--device cpu|cuda] [--n 600] [--pct-triples 1500]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))    # tools/
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import fastvilla as FV  # noqa: E402
from ens import adapters as A  # noqa: E402


def percentile_test(norm, dev, triples, seed=2):
    """Patches of norm.n uint8 values a < m < c with a at sorted positions 0..lo, c above hi, m between, so the 1st
    percentile interpolates between a and m and the 99th between m and c. Bounds from norm._percentiles vs
    np.percentile of the float32 patch itself (what normalize_robust clips with). Returns the number of mismatches."""
    import torch
    n = norm.n
    lo, hi = (int(np.floor((n - 1) * q / 100.0)) for q in (1.0, 99.0))
    rng = np.random.default_rng(seed)
    tri = np.sort(np.stack([rng.choice(256, 3, replace=False) for _ in range(triples)]), axis=1)
    counts = np.array([lo + 1, hi - lo, n - hi - 1])
    hist = np.zeros((triples, 256), np.int64)
    np.put_along_axis(hist, tri, counts[None, :], axis=1)
    bad = [0, 0]
    for i in range(0, triples, 256):
        got = norm._percentiles(torch.from_numpy(hist[i:i + 256].cumsum(1)).to(dev))
        for j, t in enumerate(tri[i:i + 256]):
            x = np.repeat(t, counts).astype(np.float32)
            for k, q in enumerate((1.0, 99.0)):
                bad[k] += np.float32(float(got[k][j])) != np.float32(np.percentile(x, q))   # the float32 clip bound
    print(f'percentile step: {triples} value triples at n={n}; bounds differing from np.percentile: '
          f'1st {bad[0]}, 99th {bad[1]}')
    return sum(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu'); ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--sabotage', action='store_true', help='divisor +1 ulp: the test must report mismatches')
    ap.add_argument('--pct-triples', type=int, default=1500, help='direct percentile test size (0: skip)')
    ap.add_argument('--render', default=os.path.join(_P.WORK, 'ensemble', 'calib343', 'm2', 'r343b.zarr'))
    ap.add_argument('--mask', default=None, help='render2.npy (rendered-area mask); default: next to --render')
    a = ap.parse_args()
    if a.device.startswith('cuda'):
        import depth_sweep as DS
        DS.gpu_guard(a.device)
    import torch
    INF = A._villa_imports()
    arr, _ = A.C.open_render(a.render)
    n, H, W = arr.shape
    rng = np.random.default_rng(1)
    r2 = np.load(a.mask or os.path.join(os.path.dirname(os.path.abspath(a.render)), 'render2.npy'))
    from scipy import ndimage
    edge = r2 & ~ndimage.binary_erosion(r2, iterations=40)
    ey, ex = np.nonzero(edge); iy, ix = np.nonzero(r2)
    p, Z = 128, 17
    patches = []
    for i in range(a.n):
        if i % 2:
            j = rng.integers(len(ey)); cy, cx = ey[j], ex[j]
        else:
            j = rng.integers(len(iy)); cy, cx = iy[j], ix[j]
        y0 = int(np.clip(cy - p // 2, 0, H - p)); x0 = int(np.clip(cx - p // 2, 0, W - p))
        d = int(rng.integers(-6, 7))
        s = n // 2 + d - Z // 2
        patches.append(np.asarray(arr[s:s + Z, y0:y0 + p, x0:x0 + p])[::-1].copy())
    # synthetic edge cases: constant, all zero, two-valued, 51 % zeros, ramps, extreme values
    syn = [np.zeros((Z, p, p), np.uint8), np.full((Z, p, p), 7, np.uint8), np.full((Z, p, p), 255, np.uint8)]
    t = np.zeros((Z, p, p), np.uint8); t[:, :, 64:] = 200; syn.append(t)
    t = rng.integers(0, 256, (Z, p, p)).astype(np.uint8); t.reshape(-1)[:int(0.51 * t.size)] = 0; syn.append(t)
    t = rng.integers(0, 256, (Z, p, p)).astype(np.uint8); t.reshape(-1)[:int(0.49 * t.size)] = 0; syn.append(t)
    syn.append(np.broadcast_to(np.arange(p, dtype=np.uint8)[None, None, :] * 2, (Z, p, p)).copy())
    t = rng.integers(100, 103, (Z, p, p)).astype(np.uint8); syn.append(t)
    t = rng.integers(0, 2, (Z, p, p)).astype(np.uint8) * 255; syn.append(t)
    patches += syn
    norm = FV.RobustNormTorch(Z * p * p)
    norm.sabotage = a.sabotage
    dev = torch.device(a.device)
    t0 = time.time()
    ref = [INF.normalize_flat_patch(q.astype(np.float32), 'tifxyz_robust') for q in patches]
    t_np = time.time() - t0
    P = torch.from_numpy(np.stack(patches)).to(dev)
    outs, fbs = [], []
    t0 = time.time()
    nfb = 0
    for i in range(0, len(patches), 16):
        X, fb = norm(P[i:i + 16]); fbs.append(fb.cpu().numpy())
        # the fast path's fallback handling: np.std of the patch on the CPU, the rest on the device
        X, _ = norm(P[i:i + 16], std_fn=lambda j, i=i: np.std(patches[i + j].astype(np.float32).reshape(-1)))
        nfb += norm.n_fallback
        outs.append(X.cpu().numpy())
    t_t = time.time() - t0
    X = np.concatenate(outs); fb = np.concatenate(fbs)
    R = np.stack(ref)
    print(f'fallback rows handled with CPU std: {nfb}')
    fb = np.zeros_like(fb)      # compare every row (fallback rows included)
    ok = ~fb
    same = (X[ok].view(np.uint32) == R[ok].view(np.uint32))
    per_patch_bitexact = same.reshape(ok.sum(), -1).all(1)
    md = float(np.abs(X[ok] - R[ok]).max()) if ok.any() else 0.0
    print(f'device {a.device}: {len(patches)} patches ({len(syn)} synthetic); fallback (CPU-recomputed) {int(fb.sum())}')
    print(f'non-fallback patches bit-identical: {int(per_patch_bitexact.sum())}/{int(ok.sum())}; '
          f'voxels identical {same.mean():.9f}; max |diff| {md:.3g}')
    print(f'time numpy {t_np / len(patches) * 1e3:.1f} ms/patch (1 core), torch {t_t / len(patches) * 1e3:.2f} ms/patch')
    bad = np.where(~per_patch_bitexact)[0][:5]
    for b in bad:
        idx = np.where(ok)[0][b]
        print('  mismatch patch', idx, 'n diff vox', int((~same.reshape(ok.sum(), -1)[b]).sum()))
    pct_bad = percentile_test(norm, dev, a.pct_triples) if a.pct_triples > 0 else 0
    sys.exit(0 if per_patch_bitexact.all() and pct_bad == 0 else 1)


if __name__ == '__main__':
    main()
