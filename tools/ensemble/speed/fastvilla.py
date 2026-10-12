"""Opt-in GPU fast path (2026-10-11) for the villa family (ink9_s42, ink9_s43, dnative) and
instrumented adapters for profiling. Nothing here changes ens/adapters.py; the validated default path is untouched.

FastVillaAdapter computes exactly what ens.adapters.VillaAdapter computes, but moves the CPU-side work to the GPU:
  * the tile's (Z, h, w) uint8 window stack is copied to the GPU once per tile;
  * the all-zero-patch filter (integral image), patch extraction (GPU slicing), the villa robust-MAD normalisation
    and the Hann overlap-add run on the GPU;
  * normalisation = an exact re-implementation of vesuvius.image_proc.intensity.normalization.normalize_robust for
    uint8 input (numpy 2.x / NEP 50 semantics): per-patch 256-bin histograms give the order statistics exactly; the
    two percentiles come from np.percentile itself on a constructed array with the same order statistics around the
    interpolation point (2026-10-12; a float64 lerp was 1 ulp off numpy 2.5.3 for 14 % of value pairs at the 1st
    percentile, 7.7 % at the 99th); clip, median, |x - median|, MAD median, subtraction and
    division run in float32 exactly as numpy does them. Patches that take numpy's MAD-fallback branch
    (scaled MAD < 1e-6, e.g. > 50 % zero padding) are normalised on the CPU by the original function;
  * overlap-add uses the same float32 multiply and add in the same patch order as the numpy code;
  * num/den come back to the CPU once per tile, and finalize() is the original.
The exactness test (tests/compare_maps.py) compares maps of both paths on the same tiles.

ProfVilla / ProfFastVilla: the same computations with timers (torch.cuda.synchronize at phase boundaries) for the
time split read | extract | norm | h2d | forward | post | stitch.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # tools/ensemble
from ens import adapters as A  # noqa: E402

T = defaultdict(float)          # global timers (seconds) and counters
NPCT = (1.0, 99.0)


@lru_cache(maxsize=1 << 16)
def _np_percentile(n, q, first, window):
    """float(np.percentile(x, q)) for a float32 array x of n values whose sorted values at positions first, first+1,
    ... are `window`. numpy reads only the two order statistics at its interpolation point (inside the window), so
    this is numpy's own result, including its interpolation arithmetic."""
    if window[0] == window[-1]:
        return float(window[0])                       # interpolating between equal values returns that value
    x = np.empty(n, np.float32)
    x[:first] = window[0]
    x[first:first + len(window)] = window
    x[first + len(window):] = window[-1]
    return float(np.percentile(x, q))


def _sync(dev):
    import torch
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


class RobustNormTorch:
    """normalize_robust(uint8 patch.astype(float32)) for a batch (B, Z, p, p) uint8 torch tensor, exact.
    Returns (normalised float32 (B, Z, p, p), fallback bool (B,)). Rows flagged fallback must be recomputed by the
    original numpy function (their values here are meaningless)."""

    sabotage = False

    def __init__(self, n):
        self.n = int(n)
        self.q = []
        for q in NPCT:
            # order-statistic window around numpy's interpolation index, which numpy computes itself (float32 in
            # some versions): +-(1 + n >> 21) around the float64 estimate always contains both interpolation points
            est = int(np.floor((self.n - 1) * (q / 100.0)))
            h = 1 + (self.n >> 21)
            self.q.append((q, max(0, est - h), min(self.n - 1, est + h + 1)))
        self.m1, self.m2 = self.n // 2 - 1, self.n // 2      # even n: mean of the two middle order statistics
        if self.n % 2:
            raise ValueError('odd patch size not implemented')

    @staticmethod
    def _ostat(cum, k):
        """k-th smallest (0-based) value per row from inclusive cumulative counts over values 0..255."""
        return (cum <= k).sum(1)

    def _percentiles(self, cum):
        """[lower, upper]: float64 (B,) tensors equal to float(np.percentile(patch as float32, q)), q in NPCT, exactly
        (see _np_percentile; cached; one small device->host copy per percentile and batch)."""
        import torch
        out = []
        for q, first, last in self.q:
            ks = torch.arange(first, last + 1, device=cum.device)
            win = (cum[:, None, :] <= ks[None, :, None]).sum(2).cpu().tolist()     # (B, K) order statistics
            out.append(torch.tensor([_np_percentile(self.n, q, first, tuple(w)) for w in win],
                                    dtype=torch.float64, device=cum.device))
        return out

    def __call__(self, P, std_fn=None):
        """std_fn(j) -> float(np.std(patch j as float32, flattened)) on the CPU, used only for MAD-fallback rows
        (numpy's float32 pairwise-sum std is not re-implemented); without it fallback rows are only flagged."""
        import torch
        B = P.shape[0]
        dev = P.device
        flat = P.reshape(B, -1)
        off = (torch.arange(B, device=dev, dtype=torch.int64) * 256)[:, None]
        hist = torch.bincount((flat.to(torch.int64) + off).reshape(-1), minlength=256 * B).reshape(B, 256)
        cum = hist.cumsum(1)
        b64 = self._percentiles(cum)                         # numpy's own percentile values (float32-exact)
        lo32, hi32 = (v.to(torch.float32) for v in b64)      # np.clip with python floats (NEP 50) -> float32 bounds
        # median of the clipped array = mean of the clipped middle order statistics (clip is monotone)
        s1 = self._ostat(cum, self.m1).to(torch.float32)
        s2 = self._ostat(cum, self.m2).to(torch.float32)
        c1 = torch.minimum(torch.maximum(s1, lo32), hi32)
        c2 = torch.minimum(torch.maximum(s2, lo32), hi32)
        med32 = (c1 + c2) / 2
        # |clip(v) - median| for each value v = 0..255 (float32), multiset weighted by the histogram
        vals = torch.arange(256, device=dev, dtype=torch.float32)[None, :].expand(B, 256)
        cv = torch.minimum(torch.maximum(vals, lo32[:, None]), hi32[:, None])
        dv = torch.abs(cv - med32[:, None])
        sd, perm = torch.sort(dv, dim=1)
        cc = hist.gather(1, perm).cumsum(1)
        d1 = sd.gather(1, self._ostat(cc, self.m1)[:, None])[:, 0]
        d2 = sd.gather(1, self._ostat(cc, self.m2)[:, None])[:, 0]
        mad32 = (d1 + d2) / 2
        smad64 = mad32.to(torch.float64) * 1.4826
        fallback = ~torch.isfinite(smad64) | (smad64 < 1e-6)
        if std_fn is not None:
            fbi = torch.nonzero(fallback).flatten().cpu().numpy()
            if len(fbi):
                lo_h, hi_h = b64[0][fbi].cpu().numpy(), b64[1][fbi].cpu().numpy()
                smad64 = smad64.clone()
                for jj, j in enumerate(fbi):              # normalize_robust's fallback branch, in python floats
                    std_preclip = float(std_fn(int(j)))
                    span = float(hi_h[jj]) - float(lo_h[jj])
                    cands = []
                    if np.isfinite(std_preclip):
                        cands.append(abs(std_preclip))
                    if np.isfinite(span):
                        cands.append(abs(span) / 2.0)
                    s = next((c for c in cands if c >= 1e-6), None)
                    if s is None or not np.isfinite(s):
                        s = 1.0
                    smad64[j] = s
                fallback = torch.zeros_like(fallback)
            self.n_fallback = len(fbi)
        smad32 = torch.where(fallback, torch.ones_like(smad64), smad64).to(torch.float32)
        if self.sabotage:            # TEST ONLY: shift the divisor by one float32 ulp (the exactness test must fail)
            smad32 = torch.nextafter(smad32, torch.full_like(smad32, float('inf')))
        X = P.to(torch.float32)
        X = torch.minimum(torch.maximum(X, lo32[:, None, None, None]), hi32[:, None, None, None])
        X = (X - med32[:, None, None, None]) / smad32[:, None, None, None]
        return X, fallback


class FastVillaAdapter(A.VillaAdapter):
    """VillaAdapter with GPU-side filter / extraction / normalisation / stitching (see module docstring)."""
    fast = True

    def load(self, device='cpu'):
        super().load(device)
        import torch
        self._norm = RobustNormTorch(self.depth_in * self.patch * self.patch)
        self._wm = torch.from_numpy(np.ascontiguousarray(self.wmap, dtype=np.float32)).to(self.device)
        if self.pre != 'tifxyz_robust':
            raise ValueError(f'fast path implements tifxyz_robust only, model uses {self.pre}')
        return self

    def _t(self, k, t0):
        if self._prof:
            _sync(self.device)
            t1 = time.perf_counter(); T[k] += t1 - t0
            return t1
        return t0

    _prof = False

    def infer(self, stack, oy, ox, origins, num, den, batch=16):
        import torch
        INF, p, dev = self.INF, self.patch, self.device
        t = time.perf_counter()
        Z, h, w = stack.shape
        S = torch.from_numpy(np.ascontiguousarray(stack)).to(dev)
        o = np.asarray(origins, np.int64).reshape(-1, 2) - np.array([oy, ox], np.int64)
        if len(o) == 0:
            return 0
        if (o < 0).any() or (o[:, 0] + p > h).any() or (o[:, 1] + p > w).any():
            raise ValueError('fast villa: a patch leaves the tile box (not expected on the villa grid)')
        t = self._t('h2d', t)
        # keep filter: any non-zero voxel in stack[:, ly:ly+p, lx:lx+p]  (== VillaAdapter.infer)
        nz = (S != 0).any(0).to(torch.int32)
        I = torch.nn.functional.pad(nz.cumsum(0).cumsum(1), (1, 0, 1, 0))
        og = torch.from_numpy(o).to(dev)
        ly, lx = og[:, 0], og[:, 1]
        cnt = I[ly + p, lx + p] - I[ly, lx + p] - I[ly + p, lx] + I[ly, lx]
        keep = (cnt > 0).cpu().numpy()
        kept = o[keep]
        t = self._t('extract', t)
        numg = torch.zeros((h, w), dtype=torch.float32, device=dev)
        deng = torch.zeros((h, w), dtype=torch.float32, device=dev)
        wm = self._wm
        ac = (torch.autocast('cuda', dtype=self.amp) if dev.type == 'cuda' else nullcontext())
        with torch.inference_mode():
            for i in range(0, len(kept), batch):
                chunk = kept[i:i + batch]
                P = torch.stack([S[:, y:y + p, x:x + p] for (y, x) in chunk])
                t = self._t('extract', t)
                def std_fn(j, chunk=chunk):                    # numpy MAD-fallback rows only: np.std on the CPU
                    y, x = chunk[j]
                    return np.std(stack[:, y:y + p, x:x + p].astype(np.float32).reshape(-1))
                X, fb = self._norm(P, std_fn=std_fn)
                T['fallback_patches'] += self._norm.n_fallback
                t = self._t('norm', t)
                with ac:
                    logits = self.model(X[:, None])
                    t = self._t('forward', t)
                    prob = INF.logits_to_probabilities(logits, image_hw=(p, p))[:, 0]
                contrib = prob * wm
                t = self._t('post', t)
                for j, (y, x) in enumerate(chunk):
                    numg[y:y + p, x:x + p] += contrib[j]
                    deng[y:y + p, x:x + p] += wm
                t = self._t('stitch', t)
        num += numg.cpu().numpy()
        den += deng.cpu().numpy()
        t = self._t('stitch', t)
        return int(len(kept))


class ProfVilla(A.VillaAdapter):
    """VillaAdapter.infer, line for line, with timers (synchronised at phase boundaries)."""

    def read_grid(self, *a, **k):
        t = time.perf_counter(); r = super().read_grid(*a, **k); T['read'] += time.perf_counter() - t
        return r

    def finalize(self, num, den):
        t = time.perf_counter(); r = super().finalize(num, den); T['finalize'] += time.perf_counter() - t
        return r

    def infer(self, stack, oy, ox, origins, num, den, batch=16):
        import torch
        INF, p = self.INF, self.patch
        dev = self.device
        t = time.perf_counter()
        keep = []
        for (y, x) in origins:
            ly, lx = y - oy, x - ox
            if stack[:, max(ly, 0):ly + p, max(lx, 0):lx + p].any():
                keep.append((y, x))

        def raw_patch(y, x):
            ly, lx = y - oy, x - ox
            raw = stack[:, max(ly, 0):ly + p, max(lx, 0):lx + p]
            pt = np.zeros((stack.shape[0], p, p), np.uint8)
            pt[:, max(-ly, 0):max(-ly, 0) + raw.shape[1], max(-lx, 0):max(-lx, 0) + raw.shape[2]] = raw
            return pt
        pool = self._pool()
        ac = (torch.autocast('cuda', dtype=self.amp) if dev.type == 'cuda' else nullcontext())
        T['extract'] += time.perf_counter() - t
        with torch.inference_mode(), ac:
            for i in range(0, len(keep), batch):
                t = time.perf_counter()
                chunk = keep[i:i + batch]
                raws = [raw_patch(y, x) for (y, x) in chunk]
                t1 = time.perf_counter(); T['extract'] += t1 - t
                if pool is not None:
                    normed = pool.map(A._villa_norm, [(r, self.pre) for r in raws])
                else:
                    normed = [A._villa_norm((r, self.pre)) for r in raws]
                t2 = time.perf_counter(); T['norm'] += t2 - t1
                img = torch.from_numpy(np.stack(normed))[:, None].to(dev)
                _sync(dev); t3 = time.perf_counter(); T['h2d'] += t3 - t2
                lg = self.model(img)
                _sync(dev); t4 = time.perf_counter(); T['forward'] += t4 - t3
                prob = INF.logits_to_probabilities(lg, image_hw=(p, p)).cpu().numpy()[:, 0]
                t5 = time.perf_counter(); T['post'] += t5 - t4
                for (y, x), pr in zip(chunk, prob):
                    ly, lx = y - oy, x - ox
                    num[ly:ly + p, lx:lx + p] += pr * self.wmap
                    den[ly:ly + p, lx:lx + p] += self.wmap
                T['stitch'] += time.perf_counter() - t5
        return len(keep)


class ProfFastVilla(FastVillaAdapter):
    _prof = True

    def read_grid(self, *a, **k):
        t = time.perf_counter(); r = super().read_grid(*a, **k); T['read'] += time.perf_counter() - t
        return r

    def finalize(self, num, den):
        t = time.perf_counter(); r = super().finalize(num, den); T['finalize'] += time.perf_counter() - t
        return r


MODES = {'orig': A.VillaAdapter, 'prof': ProfVilla, 'fast': FastVillaAdapter, 'fastprof': ProfFastVilla}


def make(name, mode='orig'):
    if A.SPECS[name]['family'] != 'villa':
        return A.make(name)
    return MODES[mode](name)
