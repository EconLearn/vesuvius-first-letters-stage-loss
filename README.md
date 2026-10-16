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
6. End to end on a second known find (PHerc0343, FINDINGS §1.7). With a rule fixed before any ink was computed, my own
   automatic surface (tracks-on fit, official flatten, my render) puts ink inside all 18 of the winner's letter boxes:
   dense_native **+0.69** [0.49, 0.89] and **+0.71** [0.51, 0.92] against two matched nulls. A depth-shuffled render
   loses. The letters were already known, so this shows the workflow can reach them, not that it finds text blind.
7. Villa ink inference is CPU-bound, and moving its normalisation to the GPU makes it **3.1× faster** on an RTX 4060
   (2.4–2.8× on an RTX 4090) with **bit-identical** maps (FINDINGS §6.1). It is opt-in:
   `tools/ensemble/speed/depth_sweep_fast.py --villa-fast`.
8. A pipeline that screens a whole 1,000-slice band for First Letters on one rented GPU, from fit inputs to
   line-scale strips for a reader, with a cost guard and resumable steps (`tools/cloud/`, FINDINGS §7). Measured on an
   RTX 4090: 1.4–2.3 hours and **$1.25–2.10 per band**, about $0.002 per cm² of surface.
9. A check of whether a surface sits on the papyrus sheet (`tools/qa/sheet_offset.py`, FINDINGS §8). The winner's
   snapped PHerc0343 meshes: 63–70% of vertices within 2 voxels of a sheet. My automatic surfaces: **26–35%**, median
   3.5–5 voxels off. Snapping (his method, now `SNAP=1` in the kit) raised per-letter contrast on his 18 letters by
   **+0.069** [0.035, 0.103] over d −4..+4, but only +0.029 (not significant) at the screen's depths.
10. The first systematic search for lead-based ink across all 21 First Letters scrolls, with no ink model
   (`tools/qa/lead_screen.py`, FINDINGS §9). A planted lead-like layer is detected down to about +50 grey levels
   (+75 on 9.362 µm scans); no scroll shows such a signature, and the two highest did not replicate.
11. A better automatic scroll axis (umbilicus) for crushed scrolls (`tools/qa/umbilicus/`).
   - VC3D's estimator assumes round windings. Stretching each slice back towards round before estimating brings its
     axis closer to the hand-made axes (Drobkov's, MIT).
   - On the m7-skeleton route of the kit, 7 held-out scrolls: median **1.77 → 1.26 mm** (p = 0.021). Random-axis
     stretching gives no gain.
   - As an opt-in C++ patch of VC3D's own `align_and_extract_umbilicus` / `vc_gen_umbilicus`, where the stretch comes
     from the sheet normals: **2.22 → 1.33 mm** (p = 0.0074). On the challenge's own umbilici it gives 0.85 → 0.61 mm,
     which is **not significant** (p = 0.11).
12. Reader v2 as an ensemble member, tested on a scroll none of the models was trained on (PHerc1447 w058/w060
   labels). Pixel AUC of the ensemble rises by **+0.021** [0.017, 0.026] (`tools/qa/w1447_*.py`). Reader v2 was
   trained on PHerc0343, so 343 cannot test it.
13. The screen's tile selection checked on known text (`tools/qa/screen_selection_eval.py`). With a 10 % budget it
   keeps tiles covering **15 of the 18** letters of the PHerc0343 winner; random picking keeps 17 %. Two alternative
   rules I expected to be better were worse (12/18, 8/18).
14. Automatic umbilici for the 9 First Letters scrolls that had none from anyone (`umbilici/`): PHerc0175A, 0175B,
   0306B, 0483A, 0483B, 0490A, 0490B, 0846A (plus its 2.403 µm volume) and 0846B.
   - They come from the published m7 normal grids, using the un-crush estimator of result 11 with 9 seeds per slice.
   - Each scroll has an `umbilicus.json` in the official schema, the raw estimates of all three methods, and a QA
     image.
   - These are automatic estimates, not hand annotations. Each slice carries an UNSURE flag; PHerc0490B has no single
     centre and should not be used without a look at its `qa.png`.

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
| `tools/ensemble/` | The PHerc0343 pipeline: 66-layer render, depth sweeps for several ink models, ensemble z-maps, matched-null scorer, stage-2 mapping of letter boxes onto an automatic surface, the tile-ranking control, and a GPU fast path for villa inference that is bit-identical to the default path (`speed/`) |
| `tools/cloud/` | The one-GPU screening pipeline: bootstrap with pinned environments, fit-input builders, fit supervisor, flatten wrapper, band screen and tile ranking, cost guard, jobs, a GPU lock, and a mocked dry run (`tests/dryrun.sh`). See `tools/cloud/RUNBOOK.md` |
| `tools/qa/` | Surface-on-sheet check against the m7 sheet prediction (with a push-off sabotage arm); model-free lead-ink screen and face-on viewer; umbilicus benchmark and the VC3D un-crush patch (`umbilicus/`); Reader v2 test on PHerc1447; tile-selection check |
| `umbilici/` | Automatic umbilici (official `umbilicus.json` schema) for the 9 eligible scrolls without one, with per-slice flags, all-method raw estimates, QA images and the pipeline (`scripts/umbkit.py`, `harness/`) |
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

### 11. PHerc0343 end to end (FINDINGS §1.7)

Inputs:
- Erwin Nieuwlaar's release `pherc343-first-letters` at `$VC_ROOT/ext/pherc343-first-letters` (or set
  `$PHERC343_REPO`). It provides his surface, the letter boxes and the `dense_native` model.
- The ink_9um weights (seeds 42 and 43), as listed in `tools/vcpaths.py`.
- The PHerc0343 CT. It is fetched on demand only when `ENS_ALLOW_FETCH=1` is set.

GPU runs refuse to start unless `ENS_GPU_LOCK_HELD=1`. Set it yourself if nothing else is using the card, or point
`GPU_LOCKED` at a lock wrapper (the contract is in the header of `tools/ensemble/sweep_locked.sh`).

```
# Step A: the ink chain on his surface, with matched nulls and a depth-shuffled sabotage
bash tools/ensemble/calibrate_343.sh $VC_ROOT/ext/pherc343-first-letters cuda

# Step B: my own surface. Fit z 9500-11000 with tracks on (section 3's recipe, input_use_tracks = true, track loss
# from step 7000, 10k steps), flatten it (section 3), then map his boxes onto it and score:
python tools/ensemble/stage2_q343.py --rel $VC_ROOT/ext/pherc343-first-letters \
    --regions $VC_WORK/ensemble/calib343/regions.json --ours <my flattened .tifxyz> \
    --out $VC_WORK/ensemble/stage2x --run --device cuda --models ink9_s42,ink9_s43,dnative \
    --depths=-6..6:2 --sab-depths=-2..2:2 --margin 256
python tools/ensemble/stage2/stage2_matched.py --stage2 $VC_WORK/ensemble/stage2x --models ink9_s42,ink9_s43,dnative
python tools/ensemble/stage2/rankctl_343.py --stage2 $VC_WORK/ensemble/stage2x
```

The villa-model sweeps can run on a faster GPU path that gives bit-identical maps: call
`tools/ensemble/speed/depth_sweep_fast.py --villa-fast` with the same arguments as `depth_sweep.py`. Check exactness on
your own setup with `python tools/ensemble/speed/tests/test_norm_exact.py --render <a render .zarr> --device cpu`
(it also has a `--sabotage` arm that must fail).

Known gaps:
- The 0009B and PHerc0841 targets of `e1_scale_test.py` need inputs that are not included here.
- `g3_flat_paired.py` expects every listed phase to exist for all three arms.
- On a 24 GB card the PHerc0343 track band (100 M points) needs the fit's fused Triton path (`FIT_SPIRAL_TRITON=1`);
  without it the fit runs out of memory.

## Data terms and scope

- Data: Vesuvius Challenge open data, CC-BY-NC 4.0. This repository contains no scroll data (no volumes, renders,
  labels or prediction images), only code and summary numbers.
- Legacy data: if you use the legacy data (Scrolls 1–4, Fragments 1–6), cite EduceLab-Scrolls: S. Parsons,
  C. S. Parker, C. Chapman, M. Hayashida, W. B. Seales, *EduceLab-Scrolls: Verifiable Recovery of Text from
  Herculaneum Papyri using X-ray CT*, arXiv:2304.02084 (2023).
- Known text only: every measurement is on publicly labelled text, namely the PHerc1447 organiser labels, the public
  PHerc0343P labels, and the PHerc0343 letter boxes published by Erwin Nieuwlaar. Nothing here reveals text from an
  unread scroll.

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
- **Erwin Nieuwlaar** (`pherc343-first-letters`, MIT): the PHerc0343 surface, the 18 letter boxes and the
  `dense_native` model used in FINDINGS §1.7. His scripts are not run or included here; only his published data and
  model are read.

## AI assistance

I used an AI coding assistant (Claude Code) to help write much of the code and analysis. I chose the questions,
directed the experiments, checked the results, and ran the blinded reading test myself. Every number in FINDINGS.md
was produced by these scripts on public data, and each one comes with a self-test or sabotage arm that could have
failed.

## Licence

MIT (see `LICENSE`). The patch in `patches/` modifies MIT-licensed code by YoussefMoNader (ink-8um-v8in).
