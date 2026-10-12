# RUNBOOK: the Vesuvius First Letters cloud kit (RunPod RTX 4090)

This folder (`tools/cloud/`) turns one rented RTX 4090 pod into a copy of the workstation pipeline. It then runs three
jobs: a timing check, stage 2 on PHerc0343, and the screening of the scrolls listed in `screen_scrolls.tsv`.

All locations follow `tools/vcpaths.py`: `$VC_ROOT` (default `~/vesuvius`) holds `data/`, `work/`, `ckpt/`, `ext/`,
`repos/`, the virtual environments and the job state (`$VC_ROOT/cloud_state`). On a pod, point `VC_ROOT` at a directory
on the persistent volume. The code comes from this repository (checkout or the tarball made by `pack_code.sh`).

**Nothing here creates accounts or keys, starts pods, or pays for anything.** Renting the pod, supplying the SSH key and
reading the price are manual steps (marked MANUAL below).

Status:
- The kit has a local dry run (`tests/dryrun.sh`) that runs all three jobs on mock outputs, plus the resume, budget-stop,
  stop-rule-fail and NOT-TESTABLE paths.
- Nothing in this repository has been run on a real pod since the port to `vcpaths` locations. See "Untested" at the end.

## What the kit contains and what it needs

Part of this repository (all paths follow `tools/vcpaths.py`; env vars override the defaults):

| what | where (env var) | used by |
|---|---|---|
| fit supervisor `quiet_super.py` (GPU-memory guards, autosave resume) | `tools/cloud/quiet/` (`QUIET_DIR`) | `lib.sh` `fit_tag`: jobs 1, 2, 3 |
| input builders `build_normals_band.py` (m7 normals, helper `m7normals.py`), `tracks_subset.py` (published tracks), `fetch_lasagna_band.py`, `umb_auto.py`, `umb_smooth.py`, `trk_loadcheck.py`, and for the winding sense `pick_sense.py`, `sense_validate.py` | `tools/cloud/quiet/` (`QUIET_DIR`) | `inputs.sh` |
| their working files: `logs/`, `out/`, `tmp/`, `cache/`, `runs/<tag>/checkpoint_fitted.ckpt`, `GPU_LOCK` | `$VC_WORK/quiet` (`QUIET_WORK`, `QUIET_RUNS`, `GPU_LOCK`) | everything above |
| official flatten wrapper `read_flatten.py CKPT OUT UMBILICUS EXPECTED_ITERATIONS` | `tools/cloud/read/` (`READ_DIR`); surfaces go to `$VC_WORK/read/<tag>/flat/<tag>.tifxyz` (`READ_WORK`) | `lib.sh` `flatten_tag`, `flat_path` |
| GPU lock wrapper `gpu_locked.sh LABEL command...` and its helper `pending_resume.py` | `tools/cloud/gpu/` (`GPU_LOCKED`) | `tools/ensemble/sweep_locked.sh`, i.e. the CUDA path of `stage2_q343.py --run` in job 2 |
| render66, depth_sweep, stage2_q343, scorers, display, `speed/fastvilla.py`; `g12lib.py`, `vlib/`, `vcpaths.py` | `tools/ensemble/`, `tools/` (`ENS_LANE`) | all jobs |
| spiral-fitting sources (`fit_spiral.py`, `flatten_spiral_checkpoint.py`, `tracks.py`, `build_track_crossings.py`, the C++ extension) | villa checkout made by `bootstrap.sh` (`SPIRAL_FIT_DIR`, default `$VILLA_DIR/spiral-fitting`) | fits, flatten, crossings |
| letter and null boxes `$VC_WORK/ensemble/calib343/regions.json` | derived by `bootstrap.sh` from the released surfaces with `tools/ensemble/calib343_regions.py` | job 2 |

Needed from outside (data, not code):
- the public downloads that `bootstrap.sh` and the builders make (villa, the PHerc0343 release, Hugging Face weights, the
  anonymous S3 CT volumes and m7 surface predictions, the published tracks and lasagna);
- a screening config: `screen_scrolls.tsv` ships as an example (PHerc0343 stage-2 band and the PHerc1447 band 1). To
  screen other scrolls, add one row per scroll (or set `SCREEN_CONFIG`); a scroll whose CT volume is not built in needs
  `vol_id`, the array shape and `voxel_um` from the public S3 listing, and `z_top_to_bottom` from the catalogue. Nothing in
  the kit assumes an 8.64 um volume: CT and m7 paths use `vol_id` as given, and the voxel size reaches the scroll spec, the
  flatten and the scale bars from `voxel_um`, so a 9.362 um volume needs only its row;
- for job 2 the existing PHerc0343 umbilicus line `$VC_DATA/spiral/PHerc0343/umbilicus.json` (and `PHerc0343trk`'s spec as
  the umbilicus fallback);
- the fit inputs of the screened datasets: built by `inputs.sh` or uploaded by `pc/upload_inputs.sh`. That includes the
  descriptors `umbilicus.json` and `spiral-scroll*.json`: when a dataset lacks them, `inputs.sh` builds the missing ones
  on the pod before normals and tracks (see "Descriptors" in job 3 below); nothing has to be prepared by hand.

In the dry run the heavy tools are replaced by `tests/mock_tools.py`.

## 0. One-time prerequisite: an SSH key

Create one if you have none (it costs nothing), and add the public key to your RunPod account settings (MANUAL):
```
ssh-keygen -t ed25519 -f "$KEY" -N ""
```
```
cat "$KEY.pub"
```

## 1. Rent the pod in the RunPod UI (MANUAL, about 5 min)

Pods → Deploy, then:

| setting | value | why |
|---|---|---|
| GPU | 1 × RTX 4090, **Community Cloud**, **On-Demand** (not Spot/Interruptible) | about $0.34/h; a spot preemption would kill a 10k-step fit |
| CUDA filter | "Additional filters → CUDA version" ≥ **13.0** (the host driver must be ≥ 580) | the pinned torch 2.12.1 is the CUDA-13 build of the reference environment. With an older driver, bootstrap falls back to the cu128 build and flags it |
| Template | **Runpod PyTorch**, picking a tag whose name ends in **ubuntu2404** | the kit brings its own Python 3.14.7 and torch, so only the OS matters. Ubuntu 24.04 provides gcc 13, which the C++23 spiral-fitting extension needs |
| Container disk | **40 GB** | OS, apt, tools |
| RAM | **≥ 32 GB** (shown in the deploy list) | a 16k-column strip render holds about 4 GB, and the next strip renders while the sweep runs |
| Volume disk | **250 GB** (persistent; note its mount path) | code, venvs (≈ 6 GB), weights (0.55 GB), datasets (2–5 GB × 8), CT cache (~20 GB per band, pruned after each scroll), maps (~3–4 GB per scroll), checkpoints (1 GB each) |
| Ports | keep the template's **TCP 22** (and HTTP 8888) | `scp`/`rsync` only work over "SSH over exposed TCP" |
| Env vars | none | |

- Check the $/h shown before deploying (RunPod also bills disk, about $0.10/GB/month while running).
- After the pod shows Running: Connect → **"SSH over exposed TCP"**. It shows a line `ssh USER@HOST -p PORT -i KEYFILE`.
- **Stop vs terminate.** A stopped pod still bills its volume (about $0.20/GB/month, roughly $1.60/day for 250 GB).
  **Terminate** once results are synced back.

## 2. Push the code and bootstrap (about 25–35 min, about $0.15–0.20)

The commands below run on your workstation. First set the connection from the SSH line, the pod's `VC_ROOT` (a directory
on the persistent volume) and the directory the code is unpacked into:
```
export POD=USER@HOST PORT=PORT KEY=KEYFILE POD_ROOT_REMOTE=/path/on/volume/vesuvius POD_CODE_REMOTE=/path/on/volume/code
```
`pc/pod_env.sh` defines `rsh "COMMAND"` (ssh to the pod with these settings; with `POD=local` it runs the command
locally, which is how the dry run tests the transfer scripts). Source it once per shell:
```
. tools/cloud/pc/pod_env.sh
```
Pack the repository's scripts (`dist/MANIFEST.tsv` lists every file with its sha256), copy them, and unpack:
```
bash tools/cloud/pc/push_code.sh
```
(Alternatively `git clone` this repository on the pod; then `POD_CODE_REMOTE` is that checkout.) Bootstrap is idempotent
and safe to re-run. Its log goes to `$VC_ROOT/bootstrap.log`:
```
rsh "VC_ROOT=$POD_ROOT_REMOTE bash $POD_CODE_REMOTE/tools/cloud/bootstrap.sh"
```
Bootstrap must end with `bootstrap done` and write `$VC_ROOT/cloud_state/BOOTSTRAP_OK`. It does the following:
- **Layout.** Creates the directories under `$VC_ROOT` and the `.POD` marker.
- **Packages.** Installs apt packages.
- **Python.** Installs uv 0.12.21 and CPython 3.14.7 under `$POD_TOOLS` (default `$VC_ROOT/pod_tools`).
- **Main venv.** Builds `.venv` from `lock/req-main.txt`, the exact freeze of the reference environment: 104 pins, torch
  2.12.1 CUDA 13.
- **Fit venv.** Builds `.venv-x2` from `lock/req-x2.txt`, layered over `.venv`.
- **villa.** Checks out villa @ `e0bbb8b4` (sparse) into `$VILLA_DIR`.
- **Native extensions.** Builds the vc_spiral extensions in `$SPIRAL_FIT_DIR` and checks they import.
- **His release.** Clones his release @ `e8370f1f` and checks its SHA256SUMS, then derives the letter/null boxes.
- **Weights.** Downloads dense_native, ink9 s42 and ink9 s43 from Hugging Face and checks their sha256 against
  `lock/models.tsv`. Hecate is fetched only with `--with-hecate`.
- **CT.** Creates the CT cache on the pod volume and fetches the volume `.zarray` files.
- **Checks.** Runs the CUDA check, the weights status and the cost guard init. A file missing from `QUIET_DIR`,
  `READ_DIR` or `GPU_LOCKED` (only possible if those variables point elsewhere) is reported as a WARNING line.

Then set the real price and budget. Read the $/h from the RunPod UI; the budget default is **$25**:
```
rsh "VC_ROOT=$POD_ROOT_REMOTE $POD_ROOT_REMOTE/.venv/bin/python $POD_CODE_REMOTE/tools/cloud/py/cost_guard.py init --rate 0.34 --budget 25 --force-rates"
```

## 3. Inputs: upload or rebuild (decided now, from a measured number)

The datasets of `screen_scrolls.tsv` have an `upload` flag: 1 means their fit inputs (normals, tracks, crossings) are
normally prepared on the workstation and pushed with `pc/upload_inputs.sh`; 0 means they are built on the pod.
PHerc0343x (z 9500–11000, the stage-2 band) is always built on the pod: published lasagna + own umbilicus + tracks.
A finished fit checkpoint can be uploaded too (`--with-ckpt TAG`); job 3 then skips that fit (saves about 1 h, about $0.35).

Why upload when possible:
- Rebuilding costs about 1–2 h of pod CPU per scroll. Most of that is the m7 normals: 8–25k CPU-s measured on the
  workstation, plus 5–25 min of track streaming.
- That CPU work competes on 6 vCPUs with rendering and villa normalisation, which are both CPU-bound.
- An upload costs no pod time and runs while jobs 1–2 run.

Measure the uplink (100 MB):
```
bash tools/cloud/pc/upload_inputs.sh --test-only
```
If it reports **≥ 20 Mbit/s**, set upload mode on the pod:
```
rsh "echo INPUT_MODE=upload >> $POD_ROOT_REMOTE/cloud_state/pod.env"
```
Then start the upload on the workstation in the background. It is resumable: re-run it after any break.
```
nohup bash tools/cloud/pc/upload_inputs.sh > upload_inputs.log 2>&1 &
```
If the uplink is below 20 Mbit/s, do nothing. The default `INPUT_MODE=rebuild` builds everything on the pod in the
background at nice 19.

## 4. Run the jobs

Each job:
- re-runs itself in its own process group, so the budget watchdog can stop exactly that job;
- writes every step to `$VC_ROOT/cloud_state/STATUS.log`, with detailed logs in `cloud_state/logs/`;
- can be re-run after any interruption, skipping steps that are done.

Start a job detached:
```
rsh "VC_ROOT=$POD_ROOT_REMOTE setsid nohup bash $POD_CODE_REMOTE/tools/cloud/jobs/job1_timing.sh > /dev/null 2>&1 < /dev/null &"
```
Monitor it. Each line shows elapsed pod-hours and estimated $:
```
rsh "tail -n 25 $POD_ROOT_REMOTE/cloud_state/STATUS.log; VC_ROOT=$POD_ROOT_REMOTE bash $POD_CODE_REMOTE/tools/cloud/jobs/inputs.sh status; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
```
Jobs 2 and 3 start the same way. Each refuses unless job 1's stop rule passed:
```
rsh "VC_ROOT=$POD_ROOT_REMOTE setsid nohup bash $POD_CODE_REMOTE/tools/cloud/jobs/job2_stage2_343.sh > /dev/null 2>&1 < /dev/null &"
```
```
rsh "VC_ROOT=$POD_ROOT_REMOTE setsid nohup bash $POD_CODE_REMOTE/tools/cloud/jobs/job3_quiet.sh > /dev/null 2>&1 < /dev/null &"
```
Knobs go in `cloud_state/pod.env`, one `VAR=value` per line:

| knob | default |
|---|---|
| `BUDGET_USD` | 25 |
| `RATE_USD_H` | 0.34 |
| `SCROLLS` | the `screen` rows of `screen_scrolls.tsv`, in file order |
| `TOP_FRAC` | 0.10 |
| `TILE` | 512 |
| `STRIP_COLS` | 16384 |
| `BATCH` | 0 = the tool default of 16; job 1 compares 16 and 48 |
| `RENDER_PROCS` | 3 |
| `ENS_NORM_WORKERS` | 6 |
| `PIPELINE` | 1: render the next strip while this one sweeps |
| `OVERLAP` | 0; 1 = start the next scroll's fit during this scroll's sweeps (try it if job 1 reports fit GPU utilisation < 60 %) |
| `PRUNE_CT` | 1 |
| `INPUT_MODE` | `rebuild` |
| `UPLOAD_SETS` | the `upload=1` rows of `screen_scrolls.tsv` (used by job 1 to decide which datasets still need building in upload mode) |
| `SENSE_MODE` | `both`: a screened band whose `z_top_to_bottom` is `-` in the config is fitted once per winding sense and the better one is kept (see "Unknown winding sense" below); `cw` / `acw` force that sense; `fail` skips such bands |
| `SENSE_WINDOW`, `SENSE_MIN_REL_DIFF` | 500, 0.02: the parameters of that decision rule |
| `SENSE_VALIDATE`, `SENSE_VALIDATE_SETS` | 0, `PHerc0343x`: with `SENSE_VALIDATE=1` job 3 first checks the decision rule on bands of known sense (see "Known-answer check") |
| `POD_GUARD_PATH`, `POD_GUARD_OFFSET_GB` | unset; see `pod/sitecustomize.py` (disk-guard shim for external scripts) |

The scrolls, their z-bands and fit tags are configuration, not code: `screen_scrolls.tsv` is an example (see its header
for the columns); edit it or point `SCREEN_CONFIG` at your own file with the same columns.

### Job 1: timing check (`jobs/job1_timing.sh`)
1. Starts the PHerc0343x input build in the background. Job 2 needs it anyway.
2. Renders a 1500 × 1540 px canvas of **his** released 343 surface around letters, layers k 19..47. This uses the
   CPU and fetches about 0.5–1 GB of CT from public S3.
3. Runs GPU sweeps at d = −2, 0, +2 on the reading side: ink9_s42 at batch 16, dense_native at batch 16 and 48,
   with GPU samples every second.
4. Fits **500 steps** on PHerc0343x at z 9750–10750. That is 1000 slices, like the RTX 4060 reference. Track dt starts
   at step 250, so both the pre-track and post-track rates are measured.
5. `timing_report.py` writes `cloud_state/timing.json`:
   - the fit rate `r_eff`, compared with the 4060's 1.83–2.07 it/s measured by the same parser on earlier logs;
   - villa seconds per valid Mpx per depth, compared with the 4060 anchors of 11.1–38.3 s/Mpx;
   - render seconds per Mpx-layer, S3 MB/s and GPU utilisation;
   - every later job re-projected in hours and dollars.

   `r_eff = 10000 / (7000/r_pre + 3000/r_post)` is the rate of the full 10k recipe, with track dt from step 7000.

**STOP RULE (pre-set): proceed only if `r_eff` ≥ 3.0 it/s.**
- On a FAIL, job 1 exits 5, writes `STOP_RULE=FAIL`, and jobs 2–3 refuse to start (`FORCE=1` overrides).
- Stop the pod unless you decide otherwise.

### Job 2: stage 2 on PHerc0343 (`jobs/job2_stage2_343.sh`)
1. Inputs for PHerc0343x:
   - the **published** lasagna band z 9300–11200 (`fetch_lasagna_band.py`);
   - **our own** band umbilicus from `umb_auto` (lasagna route) and `umb_smooth`. If that fails, it falls back to the
     existing 8100–9900 line and records the fallback;
   - the published m7 tracks for z 9500–11000, the official crossing cache and the official loader check.
2. Our tracks fit `q343x_trk`: z 9500–11000, 10k steps, tracks on, dt from 7000, `verified_patches` and `grad_mag`
   off, run by `quiet_super.py` with VRAM guards set for 24 GB.
3. The official flatten (`read_flatten.py`).
4. A union umbilicus for 343's outward-normal check: the existing line below z 9500 plus ours above. The original is
   kept as `umbilicus.json.original`.
5. `stage2_q343.py --run`, unchanged:
   - maps his 18 boxes onto **our** surface through 3D xyz and applies its coverage gate (exit 10 = NOT TESTABLE);
   - renders 66 layers;
   - runs ink9 s42, ink9 s43 and dense_native at d −6..+6 over 2 tile phases, plus the depth-shuffled sabotage at
     d −2..+2;
   - builds the ensembles, the stage-1 score and the strips.
6. `stage2_matched.py` builds the calibration's matched nulls (interline, ring, shifted boxes, 64 px context) on the
   stage-2 canvas, runs **`score_matched.py` unchanged**, and writes `$VC_WORK/ensemble/stage2x/matched/PASS.json`.
7. One crop per mapped letter goes to `$VC_WORK/ensemble/stage2x/letters/` for the observer.

**Pre-registered stage-2 PASS** (written 2026-10-11, before any stage-2 ink exists; also in `py/stage2_matched.py`):
- **PASS** iff `INK_EVIDENCED` holds for **both** dense_native **and** the 3-model z-mean at d = −2..+2, with ≥ 9
  scored letters in both matched nulls.
- `INK_EVIDENCED` is unchanged from the calibration. For the interline and ring nulls alike:
  - the per-letter difference (box minus its own null) has a 95 % bootstrap CI above 0;
  - the depth-shuffled sabotage does not: its mean is < 25 % of the real mean, or its CI includes 0.
- Fewer than 9 scored letters gives INCONCLUSIVE.
- If stage2_q343 exits 10 (fewer than half the letters covered), the result is NOT TESTABLE.
- Reported but not gating: the stage-1 three-null rule (`score.json`) and the single members.
- An observer must also read `stage2x/strips` and `letters/` and record the answer in `stage2x/observer.txt`.

### Job 3: screening (`jobs/job3_quiet.sh`)

For each scroll, in `SCROLLS` order (bands and fit tags from `screen_scrolls.tsv`):

1. **Inputs.** Waits for an upload in progress, otherwise builds what is missing. **Descriptors:** a dataset that lacks
   `umbilicus.json` or `spiral-scroll.json` gets the missing ones first (`inputs.sh`, marker `.desc_done`): `umb_auto.py`
   (label-free m7 route, the band's z0..z1, step (z1 - z0) / 14, seeds 1 2 3) writes the raw per-slice estimates and
   `umb_smooth.py` turns them into `umbilicus.json` and `spiral-scroll.json`, with vol_id, shape, voxel size and
   z-direction from the config (and the CW / ACW variants where the z-direction is `-`). Existing descriptors are never
   overwritten. The QA images (`umbilicus_qa.png`, `umbilicus_qa_core_zoom.png`) and the raw estimates go to
   `$RESULTS_DIR/<dataset>/descriptors/`. A failure is reported as `<dataset>: descriptors FAILED` and the scroll is
   skipped, like a failed normals or tracks build.
2. **Fit.** The tracks recipe (10k steps, dt from 7000). It is skipped when a final checkpoint exists, e.g. an uploaded one.
   A band with an unknown winding sense is fitted once per sense and the sense is chosen automatically (next section).
3. **Flatten.**
4. **`band_screen plan`.**
   - Splits the band into 16384-column strips with 128 px overlap.
   - Sets **one** normal sign for the whole surface from the umbilicus outward test. An ambiguous mesh exits 3 and
     the scroll is skipped.
   - Estimates CT. Measured on a previously fitted PHerc0343 band: 1060 valid Mpx (772 cm²), 66 strips, outward
     fraction 0.990, about **8.1k CT chunks ≈ 20 GB**.
5. **Screen.** dense_native at d −2, 0, +2 on the reading side, for every strip.
   - Each strip renders layers k 23..43 only, then sweeps.
   - The render is deleted after its maps and valid mask are saved.
   - The next strip renders on the CPU while this one sweeps on the GPU.
   - The cost guard runs before every strip.
6. **Select.** Each 512 px tile with ≥ 50 % valid pixels gets a **contrast** score: p95 − p50 of the depth-mean
   dense_native unit map. The top 10 % of tiles are selected.
   - `tiles.csv` keeps p50, p95, p99 and the mean for every tile, so another rule can be applied later without GPU.
   - `overview_scores.png` shows the score map, with the selected tiles in yellow.
7. **Ensemble.** On the selected tiles plus 128 px of context: renders k 19..47 with a pixel mask, then runs
   ink9 s42 + ink9 s43 + dense_native at d −6..+6 step 2 on the selected pixels, then `ensemble_maps` (z-mean,
   mean over depth).
8. **Draw.** His dimmed-CT yellow style (`display.compose`, CT at d = +1). Each cluster of adjacent tiles gets a crop
   PNG plus line-scale strips (450 px, 25 % overlap, 1 cm bar) under `$RESULTS_DIR/<SCROLL>/strips/` (default
   `$VC_OUT/cloud`). `index.csv` gives each cluster's canvas coordinates and score.
9. **Summary.** Writes `<SCROLL>/summary.json`.
10. **CT pruning.** The CT cache of the finished scroll is deleted. It is public and can be downloaded again.

Notes:
- The reading orientation of the flattened surfaces is not established. If letters look mirrored, flip the image
  left-right.

#### Unknown winding sense (`SENSE_MODE`)

Some scrolls have no z-direction in the catalogue (`z_top_to_bottom` = `-` in `screen_scrolls.tsv`). `umb_smooth.py` then
writes `spiral-scroll.json` without a sense plus `spiral-scroll.CW.json` and `spiral-scroll.ACW.json`, and the fit refuses
to start on a spec without a sense. With `SENSE_MODE=both` (the default; `inputs.sh` counts both variants as a ready
spec) job 3 does this instead, unattended:

1. Fit the band twice, each time with `--scroll-spec` pointing at one variant, into the run tags `<tag>_cw` and
   `<tag>_acw`. These are two separate steps (`<scroll>_fit_cw`, `<scroll>_fit_acw`), each behind its own cost-guard
   check; the second fit costs about as much as the first (the `fit_h` of `timing.json`).
2. `quiet/pick_sense.py` (step `<scroll>_sense`) compares them. **Rule:**
   - both fits must have a final checkpoint (`completed_iterations` >= 10000) and logged losses, else FAIL
     ("crashed or incomplete"); a non-finite loss in the window is a FAIL;
   - score of a sense = the mean total loss of the log lines `step N: loss = ...` with N > 10000 - `SENSE_WINDOW` (500
     iterations), but at least the last 3 logged values. The fit logs the loss every 200 iterations, so the default window
     is the three values at steps 9400, 9600 and 9800; raise `SENSE_WINDOW` for a smoother score. A step logged twice (a
     resumed fit) counts once;
   - the sense with the **lower** score wins;
   - FAIL as ambiguous when `|CW - ACW| / min(CW, ACW) < SENSE_MIN_REL_DIFF` (0.02, i.e. 2 %).
3. The winner becomes the canonical run, so flatten and screening run unchanged: `$QUIET_RUNS/<tag>/checkpoint_fitted.ckpt`
   is a symlink to the winner's checkpoint, `fit_<tag>.log` is a copy of its log, and the winner's variant becomes the
   dataset's `spiral-scroll.json` (the sense-less original is kept as `spiral-scroll.nosense.json`).
4. The decision (both scores, the steps used, the relative difference, the rule, the winner) is written to
   `$VC_DATA/spiral/<dataset>/sense_decision.json` and a `SENSE DECISION ...` line goes to `STATUS.log`.

On a FAIL the scroll is skipped, with `SENSE DECISION FAILED ...` and the reason in `STATUS.log` (and `sense_decision.json`
with `decision: refused`); the other scrolls continue. To go on with that scroll anyway, add `SENSE_MODE=cw` or
`SENSE_MODE=acw` to `cloud_state/pod.env` and re-run job 3: the finished fits are reused and the chosen sense is taken
without comparing. `SENSE_MODE=fail` skips such bands (the previous behaviour: a person copies a variant over
`spiral-scroll.json`). With `OVERLAP=1` such a band gets no pre-started fit.

The rule is a heuristic (the wrong sense should fit the normals and tracks worse). It has only been exercised on mock fit
logs (`tests/dryrun.sh`); run it once on a band whose sense is known before trusting a decision.

#### Known-answer check (`SENSE_VALIDATE=1`)

Before the real scrolls, job 3 can test the rule on a band whose catalogue sense is known (`SENSE_VALIDATE_SETS`, default
the public PHerc0343 stage-2 band, `z_top_to_bottom` = false, i.e. catalogue sense CW). It only runs when `SENSE_MODE=both` and
`SCROLLS` contains an unknown-sense scroll:

1. `inputs.sh ensure` for the validation dataset, then `quiet/sense_validate.py` writes two spec variants from its
   `spiral-scroll.json` with the catalogue properties set to null (villa refuses a `spiral_outward_sense` that contradicts
   them) and records the catalogue sense (ACW if z_direction_is_top_to_bottom != left_handed_coordinates, else CW).
2. Two fits of that band, tags `val_<dataset>_cw` and `val_<dataset>_acw`, each its own cost-guarded step (the 1500-slice fit
   estimate of `timing.json`); the band is not used for anything else, and no canonical run or spec is touched.
3. `pick_sense.py --expect <catalogue sense> --no-canonical`: the same rule as above. It must choose the catalogue sense.
   Otherwise (the other sense wins, or the pair is ambiguous, or a fit failed) the job logs
   `SENSE VALIDATION FAILED on <dataset> ...: ... the unknown-sense scrolls were NOT run` and **exits with code 7** before any
   scroll. Details are in `$VC_WORK/quiet/validate/<dataset>/sense_validation.json`.
4. On success the job logs `SENSE VALIDATION PASSED ...` and goes on with the scrolls. A passed check proves the rule on one
   band only; a failed one means its decisions should not be trusted on this setup (try a larger `SENSE_WINDOW`, or pick
   senses by hand with `SENSE_MODE=cw|acw`, or skip the check with `SENSE_VALIDATE=0`).

Each validated dataset costs two extra fits. Add more known-sense datasets to the config and to `SENSE_VALIDATE_SETS` to test
both senses (a `true` band has catalogue sense ACW).

### What is kept and what is pruned
- **Kept:** fits, flattened surfaces, all model maps (uint8 tif), valid masks, `dmean` (float16 npz), every JSON,
  CSV and PNG.
- **Pruned (regenerable intermediates only):** strip renders, the per-depth ensemble `.npy` (`ensemble_maps.py`
  rebuilds them from the kept maps in seconds), and the CT cache of a finished scroll.
- No job ever deletes a result.

## 5. Results back to the workstation (small files only)

Always look at the size first:
```
bash tools/cloud/pc/sync_back.sh --dry-run
```
Then sync:
```
bash tools/cloud/pc/sync_back.sh
```
- This pulls PNG, JSON, CSV, TSV, TXT and logs from `cloud_state/`, the results directory, `work/ensemble/stage2x/`,
  the timing dir and the fit-supervisor logs into `$SYNC_DST` (default `$VC_WORK/cloud/results_pod`).
- Files are capped at 60 MB each. **CT, renders, maps and npy are never pulled.**
- `--with-ckpts` adds the checkpoints (< 1 GB each); `--with-flat` adds the flattened tifxyz (about 100 MB per scroll).
- Expect about 0.2–0.4 GB of strip PNGs per scroll.

**After the final sync, terminate the pod.**

## 6. Expected durations and costs (GUESSES)

These come from an earlier cost model: 4090 = 2.5–3.5× the 4060 measured anchors, villa 3.2–15.3 s per valid Mpx per
depth, $0.34/h plus about $0.03/h disk. Job 1 replaces them with measured numbers in `timing.json`.

| step | hours (guess) | $ (guess) |
|---|---|---|
| bootstrap | 0.4–0.6 | 0.15–0.2 |
| job 1 (timing; 343x inputs stream meanwhile) | 0.5–0.9 | 0.2–0.35 |
| job 2: fit of 1500 slices (≈ 1.5× a 1000-slice fit at ≥ 2 it/s) | 1.2–1.8 | |
| job 2: render + sweeps (3 models × 17 runs × about 11 valid Mpx) + post | 0.8–2.8 | |
| **job 2 total** | **2.2–4.8** | **0.8–1.8** |
| job 3 per scroll: fit (≥ 3 it/s) + flatten | 0.9–1.2 | |
| job 3 per scroll: screen (1060 Mpx × 3 d) | 2.8–13.5 | |
| job 3 per scroll: ensemble (≈ 150 Mpx × 3 models × 7 d) | 2.8–13.2 | |
| **job 3 per scroll** | **6.7–28** | **2.5–10.5** |
| **job 3, 7 scrolls (the `timing.json` projection)** | **47–196** | **17–73** |

- **The $25 default budget covers jobs 1–2 plus about 2–6 scrolls**, depending on the villa speed job 1 measures.
- The guard stops cleanly at the budget (minus a $0.30 reserve for syncing) and keeps every result.
- After job 1, look at the measured projection and decide whether to raise the budget or narrow job 3. Options: fewer
  scrolls, a smaller `TOP_FRAC`, or the ensemble only on the top N clusters.
- **Biggest lever if job 1 shows villa sweep GPU utilisation < 50 %:** inference is CPU-bound, because robust-MAD
  normalisation runs on CPU workers and the pod has only 6 vCPUs. `VILLA_FAST=1` (the default in `jobs/lib.sh`) already
  uses the GPU fast path `tools/ensemble/speed/fastvilla.py`, which is bit-identical to the default path.

## 7. Stop rules (summary)
- Job 1: `r_eff < 3.0 it/s` → stop and report. Jobs 2–3 are locked out.
- Budget: the watchdog sends SIGTERM to the running job at budget − $0.30. The status line reads
  `BUDGET STOP …`. **The pod keeps billing until you stop it:** sync back, then stop or terminate it.
- Bootstrap failure:
  - g++ < 13: redeploy with an Ubuntu 24.04 tag;
  - driver < 580: the cu128 fallback is used and flagged in `BOOTSTRAP_OK`, and the timing numbers then come from a
    non-reference torch build;
  - a sha256 mismatch: stop.
- Idle pod: when the last job ends or stops, stop the pod. Pods bill while idle.

## 8. Files

| path | role |
|---|---|
| `bootstrap.sh` | idempotent pod setup (`--dry-run`, `--check-locks`, `--with-hecate`) |
| `pack_code.sh`, `pack_manifest.txt` | the code tarball (exact list in `MANIFEST.tsv`, written next to the tarball) |
| `screen_scrolls.tsv` | the screening configuration (example rows): datasets, scrolls, fit tags, z-bands, upload flags, CT volume ids and shapes, catalogue z-direction |
| `quiet/`, `read/`, `gpu/` | fit supervisor and input builders (incl. `pick_sense.py`, the winding-sense decision), flatten wrapper, GPU lock wrapper (see "What the kit contains") |
| `lock/req-main.txt`, `lock/req-x2.txt`, `lock/models.tsv`, `lock/PINS.txt` | exact freezes of the reference environment, weight URLs and sha256, pinned commits |
| `pod/sitecustomize.py` | pod-only shim: disk guards of external scripts check the pod volume (active only with `POD_GUARD_PATH`) |
| `py/volumes.py`, `py/r66.py`, `py/sweep.py` | the screened CT volumes added in-process; render66 and depth_sweep wrappers |
| `py/band_screen.py` | job-3 driver: plan, screen, select, ens |
| `py/stage2_matched.py` | job-2 matched-null scorer and the pre-registered PASS |
| `py/timing_report.py` | job-1 report, stop rule and projections |
| `py/cost_guard.py` | pod-hour and $ guard plus the watchdog |
| `jobs/lib.sh`, `jobs/inputs.sh`, `jobs/job{1,2,3}_*.sh` | the jobs |
| `pc/push_code.sh`, `pc/upload_inputs.sh`, `pc/sync_back.sh`, `pc/pod_env.sh` | workstation-side transfer |
| `tests/dryrun.sh`, `tests/mock_tools.py` | local dry run with mocks |

## Untested (needs the real pod)

The following were checked only by mocks or not at all:
- **The software stack on the pod:**
  - bootstrap's real installs: uv, Python 3.14.7 and the PyPI wheels on the RunPod image;
  - the CUDA-13 torch on the host driver;
  - the vc_spiral build with the pod's gcc, now done inside the villa checkout;
  - generating the letter/null boxes with `calib343_regions.py` on the pod.
- **Network sources:** HF downloads, S3 and ash2txt throughput.
- **The pipeline on the pod:**
  - the fit speed;
  - `quiet/quiet_super.py` with 24 GB thresholds;
  - the flatten;
  - `stage2_q343 --run` on CUDA through `sweep_locked.sh`;
  - the real input builders (lasagna fetch, `umb_auto`/`umb_smooth`, `tracks_subset`, `build_normals_band`) on the
    pod.
- **Modes and transfers:** `OVERLAP=1` (exercised only with mocks: the next fit is pre-started), the upload and
  sync over real SSH (tested only in local mode), the winding-sense decision and its known-answer check (mock fit logs only),
  and the on-pod descriptor build (`umb_auto.py` + `umb_smooth.py` against the real S3 data).

## Lessons from the first real pod
- Fits run with FIT_SPIRAL_TRITON=1 (jobs/lib.sh default). Without it, the inverse transform builds a points x windings
  tensor and a 100M-point band OOMed the 24 GB 4090; with it the PHerc0343x fit used 6 GB at 8-16 it/s (10k steps ~12 min).
- bootstrap installs req-x2.txt with --no-deps; before that, torchdiffeq pulled torch 2.14.1 / triton 3.8 into .venv-x2
  and shadowed the pinned torch 2.12.1. Check versions with the venv python, not the system python3.
- spiral-fitting imports the villa `vesuvius` package from `../vesuvius` (missing -> ModuleNotFoundError
  `vc3d_fiber_format`). Inside the villa checkout that directory exists; if you point `SPIRAL_FIT_DIR` elsewhere, link it.
- Watch the job with an alert on the FIRST FAIL line of `STATUS.log`, not only on job end.

## Optional: snap the flat mesh onto the sheet prediction (`SNAP=1`)
- `SNAP=1 bash jobs/job3_quiet.sh` adds a step between flatten and plan. `py/snap_meshes_nieuwlaar.py` (the PHerc0343
  winner's script, MIT, licence in its header) moves every vertex of the flat mesh along its normal to the nearest run
  of the published m7 sheet mask. `py/m7path.py` finds that prediction on the public bucket; it is read on demand,
  not stored. If the step fails, job 3 screens the unsnapped mesh and says so in STATUS.
- Effect and limits: FINDINGS §8. Use it with a wider stage-2 depth set (for example `ENS_DEPTHS=-4..4:2` or wider).
- Cost: about 10 m7 chunks per second per 8 processes; a whole band is roughly 15–40 CPU-minutes (`SNAP_PROCS`).
