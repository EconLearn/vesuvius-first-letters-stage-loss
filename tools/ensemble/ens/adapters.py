"""Model adapters for the depth sweep. Each adapter reproduces its model's released/official tiled inference on a
GLOBAL patch grid (so a canvas can be processed tile by tile with results identical to a whole-canvas run):

  family  models                 input window for depth d (N = 66 layers, d = k - 33)          grid / output
  villa   ink9_s42, ink9_s43,    17 layers S..S+16, S = N/2 + d - 8 (his run_ink.sh)            render grid, uint8
          dnative                (villa infer: robust-MAD per 128 px patch, stride 64, Hann)    = trunc(255 p)
  hecate  hecate                 16 planes at 9.6 um, centre (N-1)/2 + d, linear z interp,      9.6 um grid, uint8
                                 INTER_AREA xy resample (his hecate_prep.py); 64 px, stride 32  = rint(255 p)
  v8in    v8in (optional 5th)    24 layers N/2 + d - 12 .. +11 (offsets d-11.5..d+11.5), clip   render grid, uint16
                                 200, /255, 64 px tiles stride 21, Gaussian blend / count       = rint(65535 p)

order: 'forward' feeds layers in render order (index increasing = +d = away from the axis); 'reverse' flips the
window. In his recipe the READING side is the villa '_reverse' output (render without flip, +d outward).

load() raises WeightsMissing (with the exact download needed) when a weight file is absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from . import common as C

CKPT = C.CKPT                                   # $VC_CKPT (tools/vcpaths.py)
EXT = C.EXT_ROOT / 'weights'                    # released weights (sha256 = his ink/models.tsv); $VC_EXT/weights


class WeightsMissing(RuntimeError):
    pass


def first(*paths):
    """First existing path of the candidates (else the first candidate, so error messages name it)."""
    for p in paths:
        if Path(p).exists() and Path(p).stat().st_size > 0:
            return Path(p)
    return Path(paths[0])


SPECS = {
    'ink9_s42': dict(family='villa', ckpt=first(CKPT / 'ink_9um/hybrid_3d2d-seed42/step-075000.pth'),
                     fetch='hf scrollprize/ink_9um  hybrid_3d2d-seed42/step-075000.pth  (138 MB)'),
    'ink9_s43': dict(family='villa', ckpt=first(EXT / 'ink_9um_seed43/step-075000.pth',
                                                CKPT / 'ink_9um/hybrid_3d2d-seed43/step-075000.pth'),
                     fetch='hf scrollprize/ink_9um  hybrid_3d2d-seed43/step-075000.pth  (138,360,231 B)'),
    'dnative': dict(family='villa', ckpt=None,      # rebuilt in memory from safetensors + config (no disk copy)
                    safetensors=first(EXT / 'dense_native/dense_native_016000.safetensors',
                                      CKPT / 'dense_native/weights/dense_native_016000.safetensors'),
                    config=first(EXT / 'dense_native/train_dense_native.json',
                                 CKPT / 'dense_native/training/configs/train_dense_native.json'),
                    fetch='hf Nieuwlaar/ink9um-dense-native  weights/dense_native_016000.safetensors (272,767,904 B, '
                          'same arch as ink_9um) + training/configs/train_dense_native.json'),
    'hecate': dict(family='hecate', ckpt=first(EXT / 'hecate/hecate_9.6um.pth', CKPT / 'hecate/hecate_9.6um.pth'),
                   code=first(EXT / 'hecate/hecate.py', CKPT / 'hecate/hecate.py'),
                   fetch='hf scrollprize/hecate  hecate_9.6um.pth (522,576,086 B) + hecate.py'),
    'v8in': dict(family='v8in', ckpt=C.V8IN / 'model.safetensors', code=C.V8IN,
                 fetch='released v8-in (local)'),
    'v8in_loo': dict(family='v8in', ckpt=CKPT / 'v8in-1447-loo-w062/model.safetensors',
                     code=CKPT / 'v8in-1447-loo-w062', fetch='v8-in fine-tuned on 1447 w058+w060 (local)'),
}
ENSEMBLE4 = ['dnative', 'ink9_s42', 'ink9_s43', 'hecate']     # his four members


def _villa_imports():
    if C.VILLA_SRC not in sys.path:
        sys.path.insert(0, C.VILLA_SRC)
    from vesuvius.ink_detection.inference import infer as INF
    return INF


def _villa_norm(args):
    """uint8 (Z, p, p) patch -> villa flat normalisation (float32), as FlatPatchReader + normalize_flat_patch."""
    raw, pre = args
    INF = _villa_imports()
    return INF.normalize_flat_patch(raw.astype(np.float32), pre)


def dnative_payload(spec):
    """His rebuild_dense_native_pth.py payload, in memory: {'model': safetensors state, 'config': the training JSON
    unchanged, 'step': 16000}."""
    from safetensors.torch import load_file
    st, cfg = Path(spec['safetensors']), Path(spec['config'])
    if not st.exists() or not cfg.exists():
        raise WeightsMissing(f'dnative: need {st} and {cfg}. Download: {spec["fetch"]}')
    return {'model': load_file(str(st)), 'config': json.loads(cfg.read_text()), 'step': 16000}


def configure_from_payload(INF, payload, source):
    """villa infer.configure_model, but from an in-memory payload (same calls, same order)."""
    config = INF.InkConfig.from_mapping(payload['config'])
    if config.data.mode != 'flat':
        raise ValueError(f'{source}: flat inference requires mode=flat, got {config.data.mode!r}')
    _, state = INF.select_inference_weights(payload, source=source)
    base = INF.make_model(config)
    inc = INF.load_flat_inference_state(base, state)
    if inc.missing_keys or inc.unexpected_keys:
        raise ValueError(f'{source}: state mismatch missing={len(inc.missing_keys)} '
                         f'unexpected={len(inc.unexpected_keys)}')
    base.eval()
    cz, cy, cx = config.model.crop_size
    model = INF.TargetModel(base, input_pad_depth_to=config.model.input_pad_depth_to).eval()
    return INF.ConfiguredModel(model=model, patch_size=cy, input_depth=cz,
                               preprocessing=INF.flat_preprocessing_from_config(config.data.normalization),
                               amp_dtype=INF.resolve_amp_dtype('auto', payload, source))


class Adapter:
    family = ''
    patch = 0
    out_dtype = np.uint8
    grid_scale = 1.0           # model grid / render grid

    def __init__(self, name):
        self.name = name
        self.spec = SPECS[name]
        self.model = None
        self.device = None

    def check(self):
        p = Path(self.spec['ckpt'])
        if not p.exists() or p.stat().st_size == 0:
            raise WeightsMissing(f'{self.name}: weights not found at {p}. Download needed: {self.spec["fetch"]}')

    def grid_hw(self, Hp, Wp):
        return Hp, Wp

    def pad_in_grid(self, pr, pc):
        return pr, pc

    def window(self, d, n):
        raise NotImplementedError

    def describe(self, d, n):
        return dict(model=self.name, family=self.family, window=self.window(d, n))


# --------------------------------------------------------------------------------------------- villa (ink_9um etc.)
class VillaAdapter(Adapter):
    family = 'villa'
    out_dtype = np.uint8

    def check(self):
        if self.name == 'dnative':
            for k in ('safetensors', 'config'):
                if not Path(self.spec[k]).exists():
                    raise WeightsMissing(f'dnative: {self.spec[k]} not found. Download needed: {self.spec["fetch"]}')
            return
        super().check()

    def load(self, device='cpu'):
        import torch
        self.check()
        INF = _villa_imports()
        self.INF = INF
        if self.name == 'dnative':
            cm = configure_from_payload(INF, dnative_payload(self.spec), str(self.spec['safetensors']))
        else:
            args = INF.parse_args(['in.zarr', str(self.spec['ckpt']), 'out.tif', '--amp-dtype', 'auto',
                                   '--no-compile'])
            cm = INF.configure_model(args)
        self.device = torch.device(device)
        self.model = cm.model.to(self.device).eval()
        self.patch, self.depth_in, self.pre, self.amp = cm.patch_size, cm.input_depth, cm.preprocessing, cm.amp_dtype
        self.stride = INF.resolve_patch_stride(patch_size=self.patch, overlap=0.5, explicit_stride=None)
        self.wmap = INF.compute_importance_map_2d(patch_size=(self.patch, self.patch), mode='hann').numpy()
        return self

    def window(self, d, n):
        din = getattr(self, 'depth_in', 17)
        s = n // 2 + int(d) - din // 2
        if s < 0 or s + din > n:
            raise ValueError(f'{self.name}: d={d} needs layers {s}..{s + din - 1}, outside 0..{n - 1}')
        return list(range(s, s + din))

    def plan(self, Hm, Wm, src=None, d=None, order=None, pad=(0, 0)):
        blocks = self.INF.iter_blocks((Hm, Wm), self.patch, self.stride, None, (1, 1))
        return [(b.y0, b.x0) for b in blocks]

    def read_grid(self, src, d, order, pad, y0, y1, x0, x1):
        """Window layers in model order, padded-canvas coords [y0:y1, x0:x1] -> (Z, h, w) uint8."""
        L = self.window(d, src.n)
        st = src.read(L[0], L[-1] + 1, y0 - pad[0], y1 - pad[0], x0 - pad[1], x1 - pad[1])
        return st[::-1] if order == 'reverse' else st

    def _pool(self):
        """Robust-MAD normalisation costs ~25-45 ms per patch on one core (measured), comparable to the GPU forward,
        so on CUDA it runs in a fork pool (villa infer uses DataLoader workers for the same reason)."""
        import os
        nw = int(os.environ.get('ENS_NORM_WORKERS', '6' if self.device.type == 'cuda' else '0'))
        if nw <= 0:
            return None
        if getattr(self, '_pl', None) is None:
            import multiprocessing as mp
            self._pl = mp.get_context('fork').Pool(nw)
        return self._pl

    def infer(self, stack, oy, ox, origins, num, den, batch=16):
        """stack (Z, h, w) covers model-grid box starting (oy, ox); adds contributions of `origins` to num/den."""
        import torch
        INF, p = self.INF, self.patch
        keep = []
        for (y, x) in origins:
            ly, lx = y - oy, x - ox
            if stack[:, max(ly, 0):ly + p, max(lx, 0):lx + p].any():
                keep.append((y, x))           # villa: an all-zero raw patch contributes nothing

        def raw_patch(y, x):
            ly, lx = y - oy, x - ox
            raw = stack[:, max(ly, 0):ly + p, max(lx, 0):lx + p]
            pt = np.zeros((stack.shape[0], p, p), np.uint8)
            pt[:, max(-ly, 0):max(-ly, 0) + raw.shape[1], max(-lx, 0):max(-lx, 0) + raw.shape[2]] = raw
            return pt
        pool = self._pool()
        ac = (torch.autocast('cuda', dtype=self.amp) if self.device.type == 'cuda' else nullcontext())
        with torch.inference_mode(), ac:
            for i in range(0, len(keep), batch):
                chunk = keep[i:i + batch]
                raws = [raw_patch(y, x) for (y, x) in chunk]
                if pool is not None:
                    normed = pool.map(_villa_norm, [(r, self.pre) for r in raws])
                else:
                    normed = [_villa_norm((r, self.pre)) for r in raws]
                img = torch.from_numpy(np.stack(normed))[:, None].to(self.device)
                prob = INF.logits_to_probabilities(self.model(img), image_hw=(p, p)).cpu().numpy()[:, 0]
                for (y, x), pr in zip(chunk, prob):
                    ly, lx = y - oy, x - ox
                    num[ly:ly + p, lx:lx + p] += pr * self.wmap
                    den[ly:ly + p, lx:lx + p] += self.wmap
        return len(keep)

    def finalize(self, num, den):
        prob = np.zeros_like(num, dtype=np.float32)
        np.divide(num, den, out=prob, where=den > 1e-6)
        np.clip(prob, 0, 1, out=prob)
        prob *= 255
        return prob.astype(np.uint8)

    @staticmethod
    def to_unit(m):
        """His ensemble_maps read_map for villa uint8: clip((v - 64) / 128, 0, 1)."""
        return np.clip((m.astype(np.float32) - 64.) / 128., 0, 1)


# --------------------------------------------------------------------------------------------- Hecate 9.6 um
class HecateAdapter(Adapter):
    family = 'hecate'
    out_dtype = np.uint8
    native_um = 8.64
    target_um = 9.6
    nz = 16

    @property
    def grid_scale(self):
        return self.native_um / self.target_um

    def load(self, device='cpu'):
        import torch
        self.check()
        spec = importlib.util.spec_from_file_location('hecate_released', str(self.spec['code']))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.H = mod
        self.device = torch.device(device)
        self.model = mod.load_model(str(self.spec['ckpt']), str(self.device))
        d, p, _ = self.model.patch_size
        self.patch, self.stride = p, p // 2
        win = torch.hann_window(p, periodic=False)
        win = win[:, None] * win[None, :]
        self.win = (win / win.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001).numpy()
        return self

    def _ratio(self):
        from fractions import Fraction
        f = Fraction(self.native_um / self.target_um).limit_denominator(100)   # 0.9 -> 9/10
        return f.numerator, f.denominator

    def grid_hw(self, Hp, Wp):
        s = self.grid_scale
        return max(1, int(round(Hp * s))), max(1, int(round(Wp * s)))

    def pad_in_grid(self, pr, pc):
        return int(round(pr * self.grid_scale)), int(round(pc * self.grid_scale))

    def zpos(self, d, n):
        zc = (n - 1) / 2. + d
        step = self.target_um / self.native_um
        pos = zc + (np.arange(self.nz) - (self.nz - 1) / 2.) * step
        if pos.min() < 0 or pos.max() > n - 1:
            raise ValueError(f'hecate: d={d} needs planes {pos.min():.2f}..{pos.max():.2f}, outside 0..{n - 1}')
        return pos

    def window(self, d, n):
        return [round(float(z), 3) for z in self.zpos(d, n)]

    def plan(self, Hm, Wm, src=None, d=None, order=None, pad=(0, 0)):
        a = self.H.axis_starts
        return [(y, x) for y in a(Hm, self.patch, self.stride) for x in a(Wm, self.patch, self.stride)]

    def read_grid(self, src, d, order, pad, y0, y1, x0, x1):
        """hecate_prep on the native box covering model rows y0:y1 (aligned to the 10->9 period), then crop.
        Exact w.r.t. a whole-canvas prep only when the padded canvas H, W are multiples of 10 (checked by caller)."""
        import cv2
        num_, den_ = self._ratio()                 # 9, 10
        Hp, Wp = src.H + pad[0], src.W + pad[1]
        ny0, nx0 = (y0 // num_) * den_, (x0 // num_) * den_
        ny1, nx1 = min(Hp, -(-y1 // num_) * den_), min(Wp, -(-x1 // num_) * den_)
        pos = self.zpos(d, src.n)
        lo = np.floor(pos).astype(int); frac = (pos - lo).astype(np.float32); hi = np.minimum(lo + 1, src.n - 1)
        vol = src.read(int(lo.min()), int(hi.max()) + 1, ny0 - pad[0], ny1 - pad[0], nx0 - pad[1], nx1 - pad[1])
        base = int(lo.min())
        zs = vol[lo - base] * (1 - frac[:, None, None]) + vol[hi - base] * frac[:, None, None]
        oh = (ny1 - ny0) * num_ // den_ if (ny1 - ny0) % den_ == 0 else int(round(Hp * num_ / den_)) - ny0 * num_ // den_
        ow = (nx1 - nx0) * num_ // den_ if (nx1 - nx0) % den_ == 0 else int(round(Wp * num_ / den_)) - nx0 * num_ // den_
        out = np.stack([np.clip(cv2.resize(zs[i], (ow, oh), interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
                        for i in range(self.nz)])
        gy0, gx0 = ny0 * num_ // den_, nx0 * num_ // den_
        out = out[:, y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]
        return out[::-1] if order == 'reverse' else out    # hecate --reverse flips the (16-plane) depth

    def infer(self, stack, oy, ox, origins, num, den, batch=4, Hm=None, Wm=None):
        import torch
        p, d = self.patch, self.nz
        act, patches = [], []
        for (y, x) in origins:
            ph, pw = min(p, Hm - y), min(p, Wm - x)
            ly, lx = y - oy, x - ox
            den[ly:ly + ph, lx:lx + pw] += self.win[:ph, :pw]        # released: weight added even for empty patches
            raw = stack[:, ly:ly + ph, lx:lx + pw]
            if not raw.any():
                continue
            pt = np.zeros((d, p, p), np.uint8); pt[:, :ph, :pw] = raw
            patches.append(pt); act.append((ly, lx, ph, pw))
        with torch.inference_mode():
            for i in range(0, len(patches), batch):
                P = np.stack(patches[i:i + batch])
                raw = torch.from_numpy(P).to(self.device)
                support = raw.any(1)
                image = torch.from_numpy(P.astype(np.float32) / np.float32(self.model.divisor))[:, None].to(self.device)
                valid = support[:, None, None].expand(-1, 1, d, -1, -1)
                prob = self.model(image, valid).float().sigmoid()[:, 0]
                wt = (prob * support * torch.from_numpy(self.win).to(self.device)).cpu().numpy()
                for w_, (ly, lx, ph, pw) in zip(wt, act[i:i + batch]):
                    num[ly:ly + ph, lx:lx + pw] += w_[:ph, :pw]
        return len(patches)

    def finalize(self, num, den):
        prob = num / np.where(den > 0, den, 1)
        if not np.isfinite(prob).all():
            raise FloatingPointError('hecate: non-finite output')
        return np.rint(prob.clip(0, 1) * 255).astype(np.uint8)

    @staticmethod
    def to_unit(m):
        return m.astype(np.float32) / (65535. if m.dtype == np.uint16 else 255.)


# --------------------------------------------------------------------------------------------- v8-in (optional 5th)
class V8Adapter(Adapter):
    family = 'v8in'
    out_dtype = np.uint16
    depth_in = 24

    def load(self, device='cpu'):
        import torch
        self.check()
        d = str(self.spec['code'])
        if d not in sys.path:
            sys.path.insert(0, d)
        from ink8um import InkDetector
        from ink8um import inference as I8
        self.I8 = I8
        self.device = torch.device(device)
        self.model = InkDetector.from_pretrained(d).eval().to(self.device)
        self.patch = int(getattr(self.model, 'tile_size', 64))
        self.stride = int(getattr(self.model, 'stride', 21))
        self.clip = int(getattr(self.model, 'clip_max', 200))
        b = I8.gkern(self.patch, 1)
        self.blend = (b / b.max()).astype(np.float32)
        return self

    def window(self, d, n):
        s = n // 2 + int(d) - self.depth_in // 2
        if s < 0 or s + self.depth_in > n:
            raise ValueError(f'v8in: d={d} needs layers {s}..{s + self.depth_in - 1}, outside 0..{n - 1}')
        return list(range(s, s + self.depth_in))

    def plan(self, Hm, Wm, src=None, d=None, order=None, pad=(0, 0)):
        """Released tile_positions on the coverage mask of the whole (padded) canvas window."""
        import cv2
        L = self.window(d, src.n)
        cov = np.zeros((Hm, Wm), np.uint8)
        for r0 in range(0, src.H, 512):
            blk = src.read(L[0], L[-1] + 1, r0, min(src.H, r0 + 512), 0, src.W)
            cov[pad[0] + r0:pad[0] + r0 + blk.shape[1], pad[1]:pad[1] + src.W] = (blk.max(0) > 0) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cov = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, kernel)
        return [(y, x) for (x, y) in self.I8.tile_positions(cov, self.patch, self.stride)]

    def read_grid(self, src, d, order, pad, y0, y1, x0, x1):
        L = self.window(d, src.n)
        st = np.clip(src.read(L[0], L[-1] + 1, y0 - pad[0], y1 - pad[0], x0 - pad[1], x1 - pad[1]), 0, self.clip)
        return st[::-1] if order == 'reverse' else st

    def infer(self, stack, oy, ox, origins, num, den, batch=16):
        import torch
        import torch.nn.functional as F
        p = self.patch
        with torch.inference_mode():
            for i in range(0, len(origins), batch):
                chunk = origins[i:i + batch]
                tiles = np.stack([stack[:, y - oy:y - oy + p, x - ox:x - ox + p] for (y, x) in chunk])
                t = torch.from_numpy(np.ascontiguousarray(tiles)).float().mul_(np.float32(1 / 255.)).to(self.device)
                with torch.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                    logits = self.model(t)
                pr = torch.sigmoid(logits).float()
                if pr.shape[-2:] != (p, p):
                    pr = F.interpolate(pr, size=(p, p), mode='bilinear')
                pr = pr[:, 0].cpu().numpy()
                for q, (y, x) in zip(pr, chunk):
                    num[y - oy:y - oy + p, x - ox:x - ox + p] += q * self.blend
                    den[y - oy:y - oy + p, x - ox:x - ox + p] += 1.0
        return len(origins)

    def finalize(self, num, den):
        pr = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        return (np.clip(np.nan_to_num(pr), 0, 1) * 65535).round().astype(np.uint16)

    @staticmethod
    def to_unit(m):
        return m.astype(np.float32) / 65535.


FAMILIES = {'villa': VillaAdapter, 'hecate': HecateAdapter, 'v8in': V8Adapter}


def make(name) -> Adapter:
    if name not in SPECS:
        raise KeyError(f'unknown model {name}; known: {sorted(SPECS)}')
    return FAMILIES[SPECS[name]['family']](name)


def to_unit(name, m):
    return FAMILIES[SPECS[name]['family']].to_unit(m)


def status():
    """Which members can run now (weights present) and what to download for the rest."""
    out = {}
    for n in SPECS:
        a = make(n)
        try:
            a.check(); out[n] = 'ready'
        except WeightsMissing as e:
            out[n] = f'MISSING: {e}'
    return out


class RenderSource:
    """Zero-filled reads of a render zarr (N, H, W): read(k0, k1, r0, r1, c0, c1) in canvas coords (may be outside)."""

    def __init__(self, path, cache_rows=None):
        arr, attrs = C.open_render(path)
        self.arr, self.attrs = arr, attrs
        self.n, self.H, self.W = (int(v) for v in arr.shape)
        self._mem = None
        if cache_rows is None and self.n * self.H * self.W <= 1_500_000_000:
            self._mem = np.asarray(arr[:])              # small canvases: keep in RAM

    def shuffle_layers(self, seed):
        """SABOTAGE: permute the rendered layers (attrs rendered_k, default all) with a fixed seed."""
        rk = self.attrs.get('ensemble_render', {}).get('rendered_k', [0, self.n])
        perm = np.arange(self.n)
        perm[rk[0]:rk[1]] = rk[0] + np.random.default_rng(seed).permutation(rk[1] - rk[0])
        self.perm = perm
        self.perm_info = dict(seed=seed, rendered_k=rk, perm=perm.tolist())

    def read(self, k0, k1, r0, r1, c0, c1):
        if getattr(self, 'perm', None) is not None:
            ks = self.perm[k0:k1]
            return np.stack([self._read(int(k), int(k) + 1, r0, r1, c0, c1)[0] for k in ks])
        return self._read(k0, k1, r0, r1, c0, c1)

    def _read(self, k0, k1, r0, r1, c0, c1):
        out = np.zeros((k1 - k0, r1 - r0, c1 - c0), np.uint8)
        sr0, sr1, sc0, sc1 = max(0, r0), min(self.H, r1), max(0, c0), min(self.W, c1)
        if sr1 > sr0 and sc1 > sc0:
            src = self._mem if self._mem is not None else self.arr
            out[:, sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = np.asarray(src[k0:k1, sr0:sr1, sc0:sc1])
        return out
