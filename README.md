# First Letters workflow on known text: stage-loss budget, winding hops, and three exactness fixes

In this repository I measure where the official automatic First Letters workflow loses ink, using known text
(PHerc1447, 8.64 µm). The workflow is spiral fit → flatten → render → ink model. The repository includes the tools to
repeat every measurement. The full write-up, with every number and its CI, is in [FINDINGS.md](FINDINGS.md).

## Main results

1. The automatic spiral-fit surface hops between windings across text.
   - On the official 30k-step fit, one winding's text block lands in **18 separate 4,000-px stretches** of the
     flattened frame, and the best stretch holds **26.5%** of it.
   - On the held-out band it is 19 stretches, best 20.9%.
   - Nearest-point adherence (86–90% within 6 voxels) cannot detect this.
   - The fit's own objective prefers the hopping surface: single-winding states cost +0.75 to +1.77, and 59–67% of
     that comes from the isometry term. (I am still validating the fix; see FINDINGS §1.4.)
2. Stage-loss budget. Each figure is the AUC change when only that stage changes:
   - ink model (v8-in vs ink_9um): **+0.13**;
   - surface (automatic vs organiser): −0.04 to −0.07;
   - depth-order decision: up to −0.12;
   - render: −0.002;
   - flatten: ≈ 0.
3. Depth order. The label-free separation rule picks the wrong order on at least one winding in 5 of 7 automatic
   arms. The geometric rule is right on 100% of matched pixels: render with
   `flip-normals = not left_handed_coordinates`, then read forward.
4. The public PHerc0343P labels are mirrored left–right and offset by about 51 px. When they are registered properly,
   ink_9um scores 0.70–0.72, not below chance.
5. v8-in stitching divides by the tile count instead of the summed blend weight. This leaves a ±10–19% 21-px grid on
   every released map. There is a one-line patch in `patches/`.

## What is here

| Path | Contents |
|---|---|
| `tools/g12lib.py` | Python port of VC3D `vc_render_tifxyz` (linear path), tifxyz reader, on-demand CT chunk mirror |
| `tools/g12_validate_render.py` | Known-answer test of the renderer against the organisers' own render |
| `tools/winding_metric.py` | Winding consistency of any tifxyz surface vs reference surfaces (`--selftest` built in) |
| `tools/surf_wind.py` | Winding consistency read directly from a `fit_spiral.py` checkpoint (14 s on CPU) |
| `tools/surf_eval.py` | The fit's own loss terms at fixed checkpoints, paired with common random numbers |
| `tools/g12_g2.py` | Stage-loss scorer for ink_9um: organiser and automatic arms, sabotage arms, 3D correspondence mapping |
| `tools/g3_v8.py`, `tools/g3_need.py`, `tools/g3_score.py`, `tools/g3_flat_paired.py` | The same for v8-in (tiling phases, stitching weight field W, paired stitching test) |
| `tools/g12_fit_run.sh`, `tools/g12_flatten.py`, `tools/g12_chain.sh` | Official fit / flatten wrappers and the per-snapshot chain |
| `tools/g12_umbilicus.py`, `tools/g12_fetch_lasagna.py` | Automatic umbilicus; fetch of the published lasagna normal band |
| `tools/g12_dose_score.py`, `g12_phase_floor.py`, `g12_offset_model.py`, `g12_geom.py`, `g12_orient.py`, `g12_order_geom.py`, `g12_auto_table.py`, `surf_normals.py` | Decomposition analyses: offset dose-response, tiling-phase floor, offset model, geometry, geometric depth order, learning curve, normal bias |
| `tools/e1_scale_test.py`, `tools/e1_p343_mapping_check.py` | 1447 canvas preparation; PHerc0343P label registration and the PR-29 placement comparison |
| `tools/vlib/` | Exact AUC, block bootstrap, C2 null, elongation (`metrics.py`, with a self-test); data access; run guards |
| `tools/vcpaths.py` | All paths, from environment variables (no machine-specific paths in the code) |
| `patches/` | v8-in stitching fix (unified diff against the released `ink8um/inference.py`) |
| `results/` | The result JSONs behind every number in FINDINGS.md (summary numbers only, no scroll data) |
| `configs/PHerc1447/` | `spiral-scroll.json` used for the fits, and the automatic band-1 umbilicus |

## Install

Tested with Python 3.14.7, torch 2.12.1 (CUDA 13.0), numpy 2.5.3, on Ubuntu 24.04 (WSL2) with an RTX 4060 8 GB.

```
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

The tools use villa at commit `e0bbb8b40a2db58b1d71864f286eb85717e59e64`: `fit_spiral.py`, `flatten_spiral_checkpoint.py`,
and the `vesuvius` ink `infer`.

```
git clone https://github.com/ScrollPrize/villa $VC_ROOT/repos/villa && git -C $VC_ROOT/repos/villa checkout e0bbb8b40a2db58b1d71864f286eb85717e59e64
```

```
pip install -e $VC_ROOT/repos/villa/vesuvius --no-deps
```

The spiral-fitting tools (`g12_flatten.py`, `surf_wind.py`, `surf_eval.py`) need villa's spiral-fitting requirements,
including `pyro-ppl`. `surf_eval.py` also needs the `vc_spiral` native extensions built and a CUDA device. Point
`SPIRAL_PY` at that environment's python for the shell scripts.

Scripts that read scroll data refuse to run until you have accepted the Vesuvius Challenge data terms yourself: the
Data Agreement form, then `vesuvius.accept_terms --yes`. After that, create the marker file:

```
touch $VC_ROOT/TERMS_ACCEPTED
```

## Data layout

Nothing is redistributed here. Fetch the data from the public sources into `$VC_ROOT` (default `~/vesuvius`; see
`tools/vcpaths.py`):

```
export VC_ROOT=$HOME/vesuvius
```

```
huggingface-cli download YoussefMoNader/ink-8um-pherc1447-surfaces --repo-type dataset --revision 7e4d918712a8d8642a6fdeee45b80217c1e98317 --local-dir $VC_ROOT/data/hf/ink-8um-pherc1447-surfaces
```

```
huggingface-cli download scrollprize/ink_9um --revision 7109667e2607db1b90c37c8b09cb876ea7fe7bb1 --local-dir $VC_ROOT/ckpt/ink_9um
```

```
huggingface-cli download YoussefMoNader/ink-8um-v8in --revision d89166b41a3f5fad7749b3d7c0fdd1bd3695d844 --local-dir $VC_ROOT/ckpt/v8in
```

The renderer fetches the PHerc1447 CT chunks it needs on demand from `s3://vesuvius-challenge-open-data`. That is
about 2.4 GB for band 1, written under `$VC_ROOT/data/s3/`. `g12_fetch_lasagna.py` fetches the lasagna normal band:
867 MB for z 12900–14300.

## Reproduce

Each step names its output. Rough times are for an RTX 4060 8 GB.

### Quick self-tests (minutes)

Metrics on synthetic data:

```
python tools/vlib/metrics.py
```

The renderer against the organisers' w058 render: expect r = 0.99987, within1 = 1.0.

```
python tools/g12_validate_render.py 2950 3900 256
```

The winding metric: identity 1.000, sabotage about 0.67.

```
python tools/winding_metric.py --selftest
```

### 1. Organiser canvases and ink_9um reference (E1 prep + infer)

```
python tools/e1_scale_test.py --stage prep --targets 1447_w058 1447_w060
```

```
python tools/e1_scale_test.py --stage infer --targets 1447_w058 1447_w060 --arms A --ckpts s42
```

### 2. Fit inputs

First, fetch the lasagna band:

```
python tools/g12_fetch_lasagna.py 12900 14300
```

Pack it with villa's `spiral-fitting/pack_resident_pools.py --normal-group 2` (the villa spiral-fitting README
explains its arguments). Then copy the dataset description and the automatic umbilicus into the dataset directory:

```
cp configs/PHerc1447/spiral-scroll.json $VC_ROOT/data/spiral/PHerc1447/ && cp configs/PHerc1447/umbilicus_auto_band1.json $VC_ROOT/data/spiral/PHerc1447/umbilicus.json
```

To recompute the raw umbilicus estimates instead (8 slices × 3 seeds; FINDINGS smooths them with a Huber line):

```
python tools/g12_umbilicus.py umbilicus_raw.json 12900 14300 200 1 2 3
```

### 3. Official fit, band 1 (about 9 h 40 min)

```
bash tools/g12_fit_run.sh g1_fit30k 13100 14100 30000
```

### 4. Organiser arms (render, sabotage, dose, phases), then the automatic chain

```
python tools/g12_g2.py org --procs 6
```

```
python tools/g12_g2.py phase && python tools/g12_g2.py asub
```

```
python tools/g12_g2.py infer w058_R w060_R w058_P10 w060_P10 w058_M10 w060_M10 w058_Asub w060_Asub
```

For the dose curve, also infer `w0xx_P2 w0xx_M2 w0xx_P4 w0xx_M4 w0xx_P6 w0xx_M6`. For the tiling-phase floor, also
infer `w0xx_Rph16_16 w0xx_Rph32_32 w0xx_Rph48_48 w0xx_Rph32_0`. In both cases `w0xx` means both w058 and w060.

```
bash tools/g12_chain.sh it30000 $VC_ROOT/work/g1g2/runs/g1_fit30k/checkpoint_fitted.ckpt
```

### 5. Learning-curve table, dose-response, phase floor, geometric depth order

```
python tools/g12_auto_table.py it30000flat it30000raw
```

```
python tools/g12_dose_score.py && python tools/g12_phase_floor.py && python tools/g12_order_geom.py it30000flat
```

### 6. v8-in arms

Build the NEED masks, write a job file (lines: `ZARR NAME ORDER DR DC`, with order `rev` for organiser canvases and
`fwd` for automatic canvases), then run and score:

```
python tools/g3_need.py org && python tools/g3_need.py auto it30000flat $VC_ROOT/work/g1g2/arms
```

```
python tools/g3_v8.py run jobs.txt
```

```
G3BAND=b1 python tools/g3_score.py --c2
```

### 7. Winding consistency

In the flattened frame:

```
python tools/winding_metric.py $VC_ROOT/work/g1g2/flat/it30000.tifxyz --ref w058 --ref w060 --zband 13150 14050 --mask labels --out wm_band1.json
```

From the checkpoint (CPU, 14 s):

```
python tools/surf_wind.py $VC_ROOT/work/g1g2/runs/g1_fit30k/checkpoint_fitted.ckpt $VC_ROOT/data/spiral/PHerc1447/umbilicus.json $VC_ROOT/work/g1g2/org/w058,$VC_ROOT/work/g1g2/org/w060 wind.json --stride 2
```

### 8. The fit's own loss at fixed checkpoints (paired)

Set `FIT_SPIRAL_CONFIG_OVERRIDES` exactly as in `g12_fit_run.sh`, then:

```
python tools/surf_eval.py --dataset $VC_ROOT/data/spiral/PHerc1447 --out eval.json --mult 8 --reps 6 CKPT_A CKPT_B
```

### 9. PHerc0343P label registration

This needs `ink/unused/P343p/20250511003658/` from the `scrollprize/datasets` Hugging Face bucket, placed under
`$VC_ROOT/data/hf_bucket/scrollprize_datasets/`. It also needs the segment's 8.64 µm surface volume under
`$VC_ROOT/data/s3/PHerc0343P/segments/20250511003658-tifxyz/surface-volumes/8.64um-1.2m-116keV-volume-20250521134555.zarr`.

```
python tools/e1_scale_test.py --stage labels343 && python tools/e1_p343_mapping_check.py
```

### 10. v8-in stitching, paired test

```
python tools/g3_flat_paired.py ph0_0
```

Band 2 (the held-out replication) uses the same tools: fit z 11000–12350, its own umbilicus, and `G3BAND=b2` for
`g3_need.py` and `g3_score.py`.

Known gaps:
- The 0009B and PHerc0841 targets of `e1_scale_test.py` need inputs that are not included here.
- `g3_flat_paired.py` expects every listed phase to exist for all three arms.

## Data terms and scope

- Data: Vesuvius Challenge open data, CC-BY-NC 4.0. This repository contains no scroll data (no volumes, renders,
  labels or prediction images), only code and summary numbers.
- Legacy data: if you use the legacy data (Scrolls 1–4, Fragments 1–6), cite EduceLab-Scrolls: S. Parsons,
  C. S. Parker, C. Chapman, M. Hayashida, W. B. Seales, *EduceLab-Scrolls: Verifiable Recovery of Text from
  Herculaneum Papyri using X-ray CT*, arXiv:2304.02084 (2023).
- Known text only: every measurement is on publicly labelled text, namely the PHerc1447 organiser labels and the
  public PHerc0343P labels. Nothing here reveals text from an unread scroll.

## Credits

- **ScrollPrize/villa** (MIT): `fit_spiral.py`, `flatten_spiral_checkpoint.py`, ink `infer`.
  - My renderer and umbilicus tools re-implement, in Python, the behaviour of VC3D's `vc_render_tifxyz` and
    `align_and_extract_umbilicus`. VC3D itself is GPL-3.0; no VC3D source is included.
- **The Vesuvius Challenge organisers:** scans, organiser surfaces and labels, lasagna normals, m7 surface
  predictions, the ink_9um model.
- **YoussefMoNader:** the v8-in model and the PHerc1447 surfaces dataset.
- **AndreasHad04** (villa #1898; `villa-apple-silicon` bench): the AUC definition, the C2 block-derangement null and
  the label transfer reused in `vlib/`.
- **liliandevarieux** (`vesuvius-841`): the `eval_forme` elongation metric (PR-32) and pre-registered benchmarks,
  including the PR-29 control re-registered here.
- **millerandmuller** (`first-light-pherc0826`): the published end-to-end First Letters run used as the recipe
  reference.

## AI assistance

I used an AI coding assistant (Claude Code) to help write much of the code and analysis. I chose the questions,
directed the experiments, checked the results, and ran the blinded reading test myself. Every number in FINDINGS.md
was produced by these scripts on public data, and each one comes with a self-test or sabotage arm that could have
failed.

## Licence

MIT (see `LICENSE`). The patch in `patches/` modifies MIT-licensed code by YoussefMoNader (ink-8um-v8in).
