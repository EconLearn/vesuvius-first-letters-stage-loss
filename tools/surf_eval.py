"""SURF lane, Step 1: evaluate the spiral fit's OWN loss terms at fixed checkpoints, paired (common random numbers).

Builds the official FitContext exactly as fit_spiral.py's CLI does (same Config defaults + FIT_SPIRAL_CONFIG_OVERRIDES
+ z-range count scaling), restores each checkpoint through the official resume path, then evaluates, with NO optimiser
step and NO gradient, the terms of the G1 objective (umbilicus, sym_dirichlet, dense_normals, min_spacing) plus, as
information only, dense_spacing (grad_mag winding density, weight 0 in G1) and, if patches are loaded, patch_radius /
patch_dt (the organiser-adherence terms; NOT part of the G1 objective).

Each repeat r uses the same seed for every checkpoint (CRN), so the paired difference (b) - (a) has small variance.
Weighted values use the G1 weights from the config. Sample counts are the scaled per-step counts times --mult.

Needs SPIRAL_FIT_DIR = a villa spiral-fitting checkout (tested e0bbb8b4) with the vc_spiral native extensions
built, and a CUDA device. usage (env FIT_SPIRAL_CONFIG_OVERRIDES as for the fit):
  surf_eval.py --dataset D --out OUT.json --mult 16 --reps 8 CKPT [CKPT ...]
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import os, sys, json, argparse, time, tempfile
os.environ.setdefault('FIT_SPIRAL_TRITON', '0')
os.environ.setdefault('WANDB_MODE', 'disabled')
SF = _P.SPIRAL_FIT_DIR
sys.path.insert(0, SF)
os.chdir(SF)
import numpy as np
import torch

import fit_spiral as FS
from fit_session import conventional_input_paths, load_scroll_spec
from config import Config, FitConfig
from spiral_helpers import scale_and_split_counts
from ddp_helpers import DistributedContext
from losses import (get_patch_and_umbilicus_losses, get_symmetric_dirichlet_loss, iter_lasagna_losses,
                    get_min_spacing_loss)

KEYS = ('sample_count_patches_per_step', 'sample_count_patches_per_step_for_dt', 'sample_count_relative_winding_pcls',
        'sample_count_absolute_winding_pcls', 'sample_count_unattached_pcls_per_step', 'sample_count_tracks_per_step',
        'sample_count_dense_normal_points', 'sample_count_fiber_direction_points',
        'sample_count_winding_model_relative_pairs', 'sample_count_winding_model_density_pairs',
        'sample_count_regularisation_points', 'sample_count_shell_samples')


def make_ctx(dataset, ckpt, cache):
    scroll = load_scroll_spec(dataset, None)
    paths = conventional_input_paths(dataset, scroll)
    config = Config().as_dict()
    config.update(FS.get_env_config_overrides())
    scale_and_split_counts(config, config['z_begin'], config['z_end'], KEYS, world_size=1)
    run_dir = tempfile.mkdtemp(prefix='surf_eval_', dir=f'{_P.WORK}/surf/tmp')
    ctx = FS.FitContext(FitConfig(config), scroll=scroll, paths=paths, resume_path=ckpt, run_dir=run_dir,
                        cache_dir=cache, dist_context=DistributedContext.from_env())
    ctx.check_cuda_ready(); ctx.load_host_inputs(); ctx.resolve_output_path(); ctx.build_device_state()
    return ctx, config


@torch.no_grad()
def eval_terms(ctx, cfg, mult, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    sat = ctx.spiral_and_transform
    T = sat.get_slice_to_spiral_transform()
    dr = sat.get_dr_per_winding()
    out = {}
    pl = get_patch_and_umbilicus_losses(
        T, dr, cfg['sample_count_patches_per_step'] * mult, cfg['sample_count_patches_per_step_for_dt'] * mult,
        ctx.verified_patches_list, ctx.patch_atlas, ctx.patch_sampling_probabilities, ctx.umbilicus_zyx,
        compute_dt=True, dt_target_cache=None, crossing_map=ctx.theta_crossing_map, cfg=cfg)
    out['umbilicus'] = float(pl[1]) * cfg['loss_weight_umbilicus']
    if ctx.verified_patches_list:
        out['INFO_patch_radius'] = float(pl[0]) * cfg['loss_weight_patch_radius']
        out['INFO_patch_dt'] = float(pl[2]) * cfg['loss_weight_patch_dt']
    out['sym_dirichlet'] = float(get_symmetric_dirichlet_loss(
        T, dr, ctx.shell_outer_winding_idx, cfg['sample_count_regularisation_points'] * mult * 4,
        cfg=cfg, z_begin=ctx.z_begin, z_end=ctx.z_end)) * cfg['loss_weight_sym_dirichlet']
    n_norm = cfg['sample_count_dense_normal_points'] * mult
    vals = {}
    try:
        for name, v in iter_lasagna_losses(T, dr, ctx.lasagna_volume, ctx.shell_outer_winding_idx, n_norm,
                                           compute_spacing=True, compute_normals=True, cfg=cfg,
                                           z_begin=ctx.z_begin, z_end=ctx.z_end):
            vals[name] = float(v)
    except Exception as e:  # grad_mag store may be absent
        vals = {}
        for name, v in iter_lasagna_losses(T, dr, ctx.lasagna_volume, ctx.shell_outer_winding_idx, n_norm,
                                           compute_spacing=False, compute_normals=True, cfg=cfg,
                                           z_begin=ctx.z_begin, z_end=ctx.z_end):
            vals[name] = float(v)
        out['INFO_spacing_error'] = str(e)[:200]
    out['dense_normals'] = vals['dense_normals'] * cfg['loss_weight_dense_normals']
    if 'dense_spacing' in vals:
        out['INFO_dense_spacing_unweighted'] = vals['dense_spacing']
    ms, _ = get_min_spacing_loss(sat, ctx.shell_outer_winding_idx, cfg, ctx.z_begin, ctx.z_end, with_metrics=False)
    out['min_spacing'] = float(ms) * cfg['loss_weight_min_spacing']
    out['TOTAL_G1_objective'] = out['umbilicus'] + out['sym_dirichlet'] + out['dense_normals'] + out['min_spacing']
    out['dr_per_winding'] = float(dr)
    return out


def load_state(ctx, ckpt):
    ck = FS.load_checkpoint_cpu(ckpt)
    ctx.load_checkpoint(ck)
    return int(ck['completed_iterations'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mult', type=int, default=16); ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--cache', default=f'{_P.WORK}/g1g2/cache')
    ap.add_argument('ckpts', nargs='+')
    a = ap.parse_args()
    os.makedirs(f'{_P.WORK}/surf/tmp', exist_ok=True)
    t0 = time.time()
    ctx, cfg = make_ctx(a.dataset, a.ckpts[0], a.cache)
    print(f'context built in {time.time()-t0:.0f}s; patches={len(ctx.verified_patches_list)}', flush=True)
    res = {'dataset': a.dataset, 'mult': a.mult, 'reps': a.reps, 'overrides': FS.get_env_config_overrides(),
           'ckpts': {}}
    for ck in a.ckpts:
        it = load_state(ctx, ck)
        reps = [eval_terms(ctx, cfg, a.mult, 1000 + r) for r in range(a.reps)]
        keys = [k for k in reps[0] if isinstance(reps[0][k], float)]
        summ = {k: {'mean': float(np.mean([x[k] for x in reps])), 'sd': float(np.std([x[k] for x in reps], ddof=1))}
                for k in keys}
        res['ckpts'][ck] = {'iters': it, 'summary': summ, 'reps': reps}
        print(ck, it, json.dumps({k: round(v['mean'], 5) for k, v in summ.items()}), flush=True)
    # paired differences vs the first checkpoint
    base = res['ckpts'][a.ckpts[0]]['reps']
    for ck in a.ckpts[1:]:
        rr = res['ckpts'][ck]['reps']
        diff = {}
        for k in base[0]:
            if not isinstance(base[0][k], float):
                continue
            d = np.array([x[k] - y[k] for x, y in zip(rr, base)])
            diff[k] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d)))}
        res['ckpts'][ck]['paired_minus_first'] = diff
        print('PAIRED', ck, 'minus', a.ckpts[0], json.dumps({k: [round(v['mean'], 5), round(v['se'], 5)] for k, v in diff.items()}), flush=True)
    res['seconds'] = time.time() - t0
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
