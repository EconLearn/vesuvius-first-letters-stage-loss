#!/usr/bin/env python
"""G3: released v8-in (ink8um, stride 21, 24 layers, clip 200) on G1G2 28-layer renders, with tiling phases.

Input canvas = a G1G2/G3 arm zarr (28, H, W) uint8 rendered by our validated vc_render_tifxyz port. v8-in takes the
centred 24 layers (indices 2..25 = offsets -11.5..+11.5, exactly the organisers' 24-layer render offsets).
Depth order (a-priori, geometric rule from G1G2 s5.4): organiser-surface renders -> 'rev' (as the released maps);
automatic renders made with flip-normals -> 'fwd'. The other order is only run as a check.
Tiling phase (dr, dc): the stack is zero-padded at top/left, predicted, cropped back (G1G2 s3.2 recipe, scaled to
v8-in's stride 21: 1/4, 1/2, 3/4 of the stride on the diagonal + (1/2, 0)).

Exactness of the two speed changes (both checked by `selftest`, max |diff| must be 0):
  * prediction runs under torch.inference_mode() (the released predict_stack builds an autograd graph it never uses);
  * NEED masks: if $VC_ROOT/work/g3/need/<canvas>.npy exists (bool H x W = the canvas pixels the scorer reads),
    only the tiles that touch a needed pixel are predicted. Every tile covering a needed pixel is still run, with the
    released tile grid, blend and normalisation, so needed pixels are bit-identical; other pixels are partial and
    are marked invalid (count of tiles saved alongside as <out>.cnt.npy is NOT kept; the scorer reads needed px only).
Output: $VC_ROOT/work/g3/v8/<canvas>__<order>__ph<dr>_<dc>.tif  (uint16, p * 65535)

The job file is re-read after every job (append lines to extend the queue); the queue idles up to IDLE_S then exits.
usage: g3_v8.py run JOBFILE [--bs N] [--frac F]     JOBFILE lines: <zarr path> <canvas name> <fwd|rev> <dr> <dc>
       g3_v8.py selftest
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import sys, os, time, json, argparse
from pathlib import Path
import numpy as np, zarr, tifffile
sys.path.insert(0, _P.TOOLS)
from vlib import guards
guards.require_terms()
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, _P.V8IN_DIR)
from ink8um import InkDetector, predict_stack
from ink8um import inference as I8

OUT = Path(f'{_P.WORK}/g3/v8')
NEED = Path(f'{_P.WORK}/g3/need')
PHASES = [(0, 0), (5, 5), (10, 10), (16, 16), (10, 0)]
IDLE_S = 1800


def load_stack(zp):
    a = np.asarray(zarr.open(str(zp), mode='r')['0'])
    assert a.shape[0] in (24, 28), a.shape      # 28 = our renders (take the centred 24); 24 = organisers' render
    st = (a[2:26] if a.shape[0] == 28 else a).transpose(1, 2, 0)
    return np.ascontiguousarray(np.clip(st, 0, 200).astype(np.uint8))


def predict_need(model, stack, reverse, need=None, stride=21, batch_size=2, progress=None):
    """predict_stack (ink8um, released) with the tile list optionally restricted to tiles touching `need`."""
    tile = int(getattr(model, 'tile_size', 64))
    if reverse:
        stack = np.ascontiguousarray(stack[..., ::-1])
    H, W = stack.shape[:2]
    points = I8.tile_positions(I8.coverage_mask(stack), tile, stride)
    if need is not None and points:
        integ = np.pad(need.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        points = [(x, y) for (x, y) in points
                  if integ[y + tile, x + tile] - integ[y, x + tile] - integ[y + tile, x] + integ[y, x] > 0]
    num = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    if not points:
        return num, 0
    blend = I8.gkern(tile, 1); blend = (blend / blend.max()).astype(np.float32)
    loader = DataLoader(I8._TileDataset(stack, points, tile), batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True)
    done = 0
    with torch.inference_mode():
        for tiles, xy in loader:
            tiles = tiles.to('cuda', non_blocking=True)
            with torch.autocast(device_type='cuda', enabled=True):
                logits = model(tiles)
            probs = torch.sigmoid(logits).float()
            if probs.shape[-2:] != (tile, tile):
                probs = torch.nn.functional.interpolate(probs, size=(tile, tile), mode='bilinear')
            probs = probs[:, 0].cpu().numpy()
            for p, (x, y) in zip(probs, xy.numpy()):
                num[y:y + tile, x:x + tile] += p * blend
                cnt[y:y + tile, x:x + tile] += 1.0
            done += len(probs)
            if progress is not None:
                progress(done, len(points))
    pred = np.divide(num, cnt, out=np.zeros_like(num), where=cnt != 0)
    return np.clip(np.nan_to_num(pred), 0, 1), len(points)


def predict(model, st, order, dr, dc, bs, need=None, prog=None):
    if dr or dc:
        st = np.pad(st, ((dr, 0), (dc, 0), (0, 0)))
        if need is not None:
            need = np.pad(need, ((dr, 0), (dc, 0)))
    p, n = predict_need(model, st, order == 'rev', need, batch_size=bs, progress=prog)
    return p[dr:, dc:], n


def outpath(name, order, dr, dc):
    return OUT / f'{name}__{order}__ph{dr}_{dc}.tif'


def read_jobs(jf):
    return [l.split() for l in Path(jf).read_text().splitlines() if l.strip() and not l.startswith('#')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('jobfile', nargs='?')
    ap.add_argument('--bs', type=int, default=2); ap.add_argument('--frac', type=float, default=0.42)
    ap.add_argument('--need-mb', type=int, default=2500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = guards.gpu_check(a.need_mb, wait_s=16 * 3600, max_util=90)
    print('gpu at start', s, flush=True)
    torch.cuda.set_per_process_memory_fraction(a.frac)
    model = InkDetector.from_pretrained(_P.V8IN_DIR).eval().cuda()
    if a.cmd == 'selftest':
        st = load_stack(f'{_P.WORK}/g1g2/arms/w060_R.zarr')[300:700, 300:900]
        need = np.zeros(st.shape[:2], bool); need[150:230, 200:330] = True
        t = time.time()
        p0 = predict_stack(model, np.ascontiguousarray(st), reverse=True, stride=21, batch_size=2, num_workers=0, device='cuda')
        t0 = time.time() - t
        t = time.time(); p1, n1 = predict(model, st, 'rev', 0, 0, a.bs); t1 = time.time() - t
        t = time.time(); p2, n2 = predict(model, st, 'rev', 0, 0, a.bs, need=need); t2 = time.time() - t
        st5 = np.pad(st, ((5, 0), (5, 0), (0, 0)))
        p5 = predict_stack(model, st5, reverse=True, stride=21, batch_size=2, num_workers=0, device='cuda')[5:, 5:]
        p6, _ = predict(model, st, 'rev', 5, 5, a.bs, need=need)
        print(json.dumps(dict(full_vs_released=float(np.abs(p1 - p0).max()),
                              need_vs_released_on_need=float(np.abs(p2 - p0)[need].max()),
                              phase5_need_vs_released_on_need=float(np.abs(p6 - p5)[need].max()),
                              phase5_vs_phase0_mean_abs=float(np.abs(p5 - p0)[need].mean()),
                              tiles_full=n1, tiles_need=n2, t_released=t0, t_full=t1, t_need=t2,
                              peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9)), flush=True)
        return
    idle_since = None; cache = {}
    while True:
        todo = [j for j in read_jobs(a.jobfile) if not outpath(j[1], j[2], int(j[3]), int(j[4])).exists()]
        if not todo:
            if idle_since is None:
                idle_since = time.time(); print('queue idle', time.strftime('%H:%M:%S'), flush=True)
            if time.time() - idle_since > IDLE_S:
                break
            time.sleep(30); continue
        idle_since = None
        job = None
        for j in todo:                      # claim a job with an atomic lock file (several workers may share a queue)
            opj = outpath(j[1], j[2], int(j[3]), int(j[4]))
            lk = opj.with_name(opj.name + '.lock')
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
                job = j; break
            except FileExistsError:
                try:
                    pid = int(lk.read_text() or 0)
                    os.kill(pid, 0)
                except (ValueError, ProcessLookupError, FileNotFoundError):
                    lk.unlink(missing_ok=True)      # stale lock of a dead worker: retry next round
                except PermissionError:
                    pass
        if job is None:
            time.sleep(10); continue
        zp, name, order, dr, dc = job[:5]
        dr, dc = int(dr), int(dc)
        op = outpath(name, order, dr, dc)
        if zp not in cache:
            cache = {zp: load_stack(zp)}
        st = cache[zp]
        nf = NEED / f'{name}.npy'
        need = np.load(nf) if nf.exists() else None
        if need is not None:
            assert need.shape == st.shape[:2], (need.shape, st.shape)
        t0 = time.time(); last = [0]

        def prog(d, n):
            if time.time() - last[0] > 300 or d == n:
                last[0] = time.time(); print(f'  {op.name} {d}/{n} tiles {time.time() - t0:.0f}s', flush=True)
        p, n = predict(model, st, order, dr, dc, a.bs, need, prog)
        tmp = op.with_name(op.name + '.part.tif')
        tifffile.imwrite(tmp, (p * 65535).round().astype(np.uint16), compression='zlib')
        tmp.rename(op)
        op.with_name(op.name + '.lock').unlink(missing_ok=True)
        print('DONE', op.name, st.shape, 'tiles', n, 'need' if need is not None else 'full', round(time.time() - t0), 's',
              'peak %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print('QUEUE EMPTY', flush=True)


if __name__ == '__main__':
    main()
