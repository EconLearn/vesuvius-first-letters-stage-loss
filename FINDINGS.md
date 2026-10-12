# Where the official First Letters workflow loses ink on known text, and why the automatic spiral fit hops windings

This is my write-up for the October 2026 Vesuvius Challenge progress prize.

Each number is labelled [m] if I measured it, [p] if it comes from a proxy measurement, or [g] if it is a guess.
Unless a row says otherwise, every AUC is pooled over PHerc1447 w058 + w060, band 1 (4,156,423 labelled pixels). CIs
are paired 64-px block bootstraps (1,000 resamples).

## Main results

All results are on known text: PHerc1447 at 8.64 µm. In my earlier survey of the eligible list, that was the scan type
of 9 of the 22 First-Letters-eligible volumes (I did not re-check this for this package). The main results replicate
on a second, held-out z-band.

1. The automatic spiral-fit surface hops between windings across text, and the fit's own loss prefers the hopping.
   - On the official 30k-step fit, w058's text block is spread over **18 separate 4,000-px stretches** of the
     automatic flattened frame. The best stretch holds **26.5%** of the text. For w060 it is 5 stretches, best 47.9%.
     On band 2 (w060) it is 19 stretches, best 20.9% [m].
   - Nearest-point adherence hides this: it reports 86–90% of the area within 6 voxels, and ranks band 2 as the
     *best* surface [m].
   - The diagnosis is an objective error rather than a search error. Every single-winding state I built scores worse
     on the fit's own objective, by **+0.75 to +1.77**. Of that penalty, 59–67% comes from the symmetric-Dirichlet
     isometry term, measured against a constant-pitch reference spiral [m].
   - This means that "no letters seen" on an automatic surface of an unread scroll can be uninformative.
   - The fix (§1.4) is to turn tracks on, as the published PHerc0826 run did. Ink AUC goes 0.785 → 0.802 (+0.017 [+0.007, +0.028]), and text contiguity doubles (w058 0.27 → 0.51, w060 0.49 → 0.90). One problem is still open: the from-scratch tracks fit doubles the winding density.
     - Tracks keep the text within about 1 fitted winding on both bands: spread90 1.06 / 0.84 / 1.26, against 7–16 for
       every isometry and normals lever.
     - The ×1.15 tracks sabotage loses (share 0.89 → 0.14).
     - The pre-registered switch count is invalid near a half-winding phase (synthetic test), and I report it as such.
2. This is the first stage-by-stage loss budget of the workflow on known text, as far as I know (I found none
   published in villa issues or the Substack as of 2026-10-05). Each row changes one stage on identical pixels.

   | Stage | Change measured | Effect |
   |---|---|---|
   | Ink model | v8-in instead of ink_9um | **+0.13** [m] |
   | Surface | automatic surface instead of the organiser surface | **−0.04 to −0.07**, replicated [m] |
   | Depth order | the label-free order decision | up to **−0.12** [m] |
   | Tiling phase (ink_9um) | position of the inference tile grid | SD 0.010 [m] |
   | Render | my renderer instead of the organisers' | −0.002 [m] |
   | Flatten | flattened instead of raw surface | ≈ 0 [m] |
3. Depth order has a free fix. The workflow's label-free "separation" rule picked the wrong order on at least one
   winding in 5 of 7 automatic arms. The pooled cost was −0.008 to −0.121, and up to −0.175 on one winding [m].
   - A geometric rule is right on 100% of matched pixels: render with `flip-normals = not left_handed_coordinates`,
     then read forward.
   - Under v8-in the wrong order scores chance (0.4965) [m].
4. The public PHerc0343P control labels are mirrored left–right and offset by about 51 px from the public 8.64 µm
   volume.
   - Once the labels are registered correctly, ink_9um scores 0.696 / 0.723 [m].
   - I reproduce 0.406 with the placement used in liliandevarieux/vesuvius-841 PR-29 (PR-29 reported 0.418).
5. The released v8-in stitching divides by the tile count instead of the summed blend weight.
   - This leaves a ±10–19% multiplicative 21-px grid on every released map [m].
   - A one-line patch fixes it. The AUC gain is +0.0008 [+0.0007, +0.0009], which is tiny, but the grid shows up as
     lines on low-signal crops.
6. Pixel AUC does not predict human legibility (§1.5).
   - In a blinded test with 33 audited catch crops, isolated letter crops of v8-in maps gave 0/14 correct on
     automatic surfaces and 1/14 on the organiser surface (AUC 0.84), with 0 false alarms [m].
   - At line scale the same letters are partly recognisable. Reading protocols need line-scale context and a piloted
     positive control.
   - No automatic statistic located text regions (§1.6, pre-registered, held-out sheets). Labelled text is not even
     inkier than unlabelled papyrus at window scale (v8in AUC 0.25 / 0.19). Text-blind detectors can pass common
     nulls [m].
   - End to end on PHerc0343 (§1.7): my own automatic surface puts ink inside all 18 of the winner's letter boxes
     (dense_native +0.69 [0.49, 0.89] / +0.71 [0.51, 0.92] against matched nulls; a depth-shuffled render loses) [m].
7. Tools (MIT licence). All of them have self-tests and sabotage arms, and I reran this package's copies on my PC.
   - A Python port of `vc_render_tifxyz`: correlation 0.99987, with 100% of voxels within 1 grey level of the
     organisers' render [m].
   - Two winding-consistency metrics: one in the flattened frame, and one read directly from a checkpoint (14 s on CPU).
   - A paired evaluator for the fit's own loss.
   - A stage-loss scorer.
   - A faster path for villa ink inference: 3.1× on an RTX 4060 and 2.4–2.8× on an RTX 4090, with maps bit-identical
     to the default path (§6.1) [m].

## Setup

| Item | Setting |
|---|---|
| Scroll and labels | PHerc1447, volume `20250521151220-8.640um-1.2m-116keV-masked.zarr`. Organiser surfaces w058/w060 (w062 for development only), with their public ink labels ∩ certainty mask. Source: HF `YoussefMoNader/ink-8um-pherc1447-surfaces` @7e4d9187 |
| Bands | Band 1: fit z 13100–14100, scored on z 13150–14050 (w058 2.93M px, w060 1.22M px). Band 2 (held out, never used for tuning): fit z 11000–12350, scored on w060 z 11050–12300 (1.76M px) |
| Fit | Official villa `fit_spiral.py` (e0bbb8b4), 30,000 steps, tracks off, automatic umbilicus (Python port of VC3D `align_and_extract_umbilicus`), published lasagna normals. Then the official `flatten_spiral_checkpoint.py` |
| Render | 28 layers via my `vc_render_tifxyz` port; automatic renders use `flip-normals` per the catalogue |
| Ink models | ink_9um seed42 step-075000 (`scrollprize/ink_9um` @7109667e) via villa `infer` defaults, `--direction both`. v8-in base (`YoussefMoNader/ink-8um-v8in` @d89166b4): stride 21, 24 centred layers |
| Scoring | Exact histogram AUC (the same definition as AndreasHad04's `xs_score.py`). Predictions are mapped to the organiser label grid by 3D nearest-surface correspondence, with coverage cap 14 vox; uncovered pixels score 0 |
| Compute | One RTX 4060 8 GB, shared. Fit: 9 h 40 min for 30k steps (eager CUDA). Flatten: 6 min. Render: 8–9 min per snapshot (CPU). ink_9um inference: 41 min per snapshot. Winding metric from a checkpoint: 14 s on CPU [m] |

## 1. The automatic surface hops windings over text

### 1.1 Measurement

I used two instruments:
- `winding_metric.py` (flattened frame) maps each labelled organiser pixel to the "same-sheet" point of the candidate
  surface (normal distance ≤ 14 vox, tangential distance ≤ 1.5 grid steps) and measures how fragmented the text is in
  the candidate's own frame.
- `surf_wind.py` (checkpoint-native) reads the fitted winding number of each organiser pixel straight from the
  checkpoint.

| Surface | densest_share (text in the best 4,000-px stretch) | n_stretches | switch prob. per 300 px | old adherence ≤ 6 vox | fitted windings holding the text (checkpoint metric) |
|---|---|---|---|---|---|
| Band 1, w058 | **0.265** | **18** | 0.285 | 0.869 | 14 (share 0.280; 6.2 switches per 1,000 px) |
| Band 1, w060 | 0.479 | 5 | 0.283 | 0.861 | 4 (share 0.473) |
| Band 2, w060 | **0.209** | **19** | 0.831 | **0.901** | — |
| Self-test, identity (organiser vs itself) | 1.000 | 1 | 0.000 | — | — |
| Self-test sabotage (500-px blocks shuffled) | 0.670 | 2 | 0.230 | — | — |

All rows [m]. The switch probability is the cap-3-vox version.

- A second instrument (pixel-level correspondences) agrees: 0.265 over 16 windows and 0.408 over 5 [m].
- Neighbouring pixels 8 px apart jump by more than 500 px on 2.6–2.8% of pairs (w058). That is roughly one sheet
  switch per ~300 px, so a 500-px letter is cut once or twice [p].
- The old adherence instrument rates band 2 best (90% within 6 vox), yet band 2 is the most fragmented surface.
  Nearest-point adherence lets any part of a crumpled surface claim a reference point.
  - A random-sign perturbation also raised it, from 26.1% to 28.9% within 2 vox [m].
  - Surfaces should not be ranked by nearest-point adherence.

### 1.2 Mechanism: drift along the sheet, not up the z axis

- The drift runs along the sheet, not along z.
  - w058's fitted winding coordinate is U-shaped along the sheet and nearly flat along z. The z-drift is about 0
    windings per 1,000 slices at 30k steps, while the text block spans 14 fitted windings along the sheet [m].
  - Band 2 adds 4.6 windings per 1,000 slices of z-drift, worst near the band edges [m].
- The fit's normals are biased, but its input normals are not.
  - The fitted in-plane normal differs from the organiser normal by −5.9° … +9.0°, changing systematically from one
    end of the sheet to the other.
  - The lasagna target normal (the fit's own input) agrees with the organiser normal to under 1° median in every
    column bin [m].
  - So the fitted windings are rounder than the crushed true sheets.
- The z-varying 2×2 linear map hardly moves. It stays near identity (aspect 1.04–1.10), while the scroll's cross
  section is a lens (aspect 1.52–1.66) [m].
- The learning curve plateaus. w058's single-winding share goes 0.087 → 0.242 → 0.268 → 0.280 at steps
  1k / 5k / 10k / 30k [m].

### 1.3 Diagnosis: the objective prefers hopping (Step 1)

I built states that keep each organiser sheet on one fitted winding, then scored them on the fit's own objective.
Method details:
- Paired with common random numbers: 8× the sample count, 6 repeats.
- Terms at the G1 weights.
- Organiser labels were used only for this diagnostic state, never in a method.

| State | w058 share | w060 share | Δ total loss vs (a) [paired SE] | Δ isometry (sym. Dirichlet) | Δ normals | Δ normals, residual capped at 10° |
|---|---|---|---|---|---|---|
| (a) official 30k fit | 0.280 | 0.473 | — (2.541 absolute) | | | |
| (a+) +3k steps, search-budget control | 0.314 | 0.611 | −0.122 ± 0.007 | −0.057 | −0.064 | −0.024 |
| (b) warm-started on the organiser sheets, 3k steps | **0.998** | **1.000** | **+1.769 ± 0.029** | **+1.042** | +0.728 | +0.027 |
| (d) (b) released for 3k steps without the organiser sheets | 0.807 (falling) | 0.997 | **+0.752 ± 0.020** | **+0.504** | +0.249 | **−0.035 ± 0.001** |

All rows [m].

What this shows:
1. Capacity is not the problem. The same flow lattice holds both sheets on single windings (b).
2. Extra search budget helps little (a+).
3. The objective is the problem.
   - The single-winding states cost more on the fit's own loss.
   - Once released (d), w058 drifts back toward hopping as the loss falls: 0.998 → 0.807 in 3k steps.
4. The isometry term carries most of the penalty: 59% at (b) and 67% at (d).
   - Mechanism [p]: the symmetric-Dirichlet term demands that fitted winding k have the canonical length
     2π(k+½)·dr of a constant-pitch spiral. Crushed real windings do not match that length, so drifting across
     sheets relieves the strain.
5. The rest comes from the outlier tail of the normals term. With that residual capped at 10°, the data term prefers
   the single-winding state (−0.035, about 40 SE). Once the outliers are capped, the data favour the true sheets.

### 1.4 The fix: tracks on (improves reading; density doubling still open)

I wrote these pre-registered bars before scoring any lever:
- PASS on a band: text share ≥ 0.70 and ≤ 1.0 switch per 1,000 px on every scored winding of that band.
- Lever called: it passes band 1 and replicates on band 2, against its paired control.
- Sabotage: a deliberately mis-specified version (for example tracks scaled ×1.15) must lose by ≥ 0.15 share.
- Guard: the median normal angle to the organiser surface must not worsen by more than 2°.

I found a problem with the switch-count instrument after scoring, so this correction is post hoc. The switch count
rounds each pixel's continuous fitted-winding coordinate v to the nearest winding. Where a sheet sits about half a
winding from the fitted windings, pixel-scale jitter in v flips that rounding back and forth.
- Synthetic test on the w058 text mask, with jitter at the level measured on the tracks fit (0.14 windings):
  - a hop-free sheet at phase 0.5 scores share 0.50 and 61.8 switches per 1,000 px;
  - the same sheet at phase 0 scores 1.00 and 0.10;
  - real hops every 800 px score 0.54 and 0.77 [m].
- So the switch bar is invalid near phase 0.5, and it under-counts real hops.
- I therefore also report spread90: the 5th–95th percentile range of v over the text, in windings. Under one winding
  means the text stays on one fitted sheet, whatever the phase. On the official 30k fit, w058's v drifts
  68 → 62 → 76 along the sheet (spread90 10.6), and this statistic catches that [m].

| Lever (3k-step warm starts unless marked) | Band 1 w058 share / switches / spread90 | Band 1 w060 share / spread90 | Band 2 w060 share / spread90 | Sabotage | Verdict |
|---|---|---|---|---|---|
| control: +3k steps (a+) | 0.314 / 6.6 / 10.6 | 0.611 / 2.0 | 0.179 / 12.5 | — | — |
| robust normals (10° cap) | 0.264 / 8.3 / 11.4 | 0.604 / 1.7 | 0.273 / 11.2 | — | no gain |
| isometry weight 10 → 2 | 0.335 / 5.0 / 7.9 | 0.565 / 1.3 | 0.215 / 9.8 | — | no gain |
| isometry weight 10 → 0.5 | 0.309 / 5.5 / 7.1 | 0.514 / 1.3 | 0.214 / 9.0 | — | no gain |
| isometry 2 + normals cap | 0.457 / 4.9 / 9.2 | 0.907 / 0.9 | 0.303 / 8.5 | — | partial on band-1 w060 only; no replication |
| per-turn intrinsic isometry reference | 0.279 / 7.0 / 10.3 | 0.551 / 2.3 | not run | — | no gain |
| normals weight ×3; fine-lattice LR ×5 | 0.341 / 5.0 / 8.3; 0.261 / 7.4 / 10.5 | 0.570; 0.614 | —; 0.185 / 12.3 | — | no gain |
| **tracks on** (m7-derived) | **0.886** / 5.0 / **1.06** | **0.943** / **0.84** | **0.717** / 14.8 / **1.26** | **×1.15 tracks: 0.136 / 0.238** (spread90 11.0 / 9.0) | **passes the share bar on both bands, and the sabotage loses by > 0.7; fails the switch bar, which is invalid here** |
| tracks on, from scratch, 10k steps (7k of 10k) | 0.518 / 14.4 / **1.26** (phase 0.48) | 0.880 / 1.08 | queued | — | on one sheet at phase ≈ 0.5 (see below) |
| from scratch, 10k: baseline / affine init / coarse→fine / 12-vox lattice | 0.263 / 15.9; 0.314 / 8.1; 0.238 / 14.4; 0.201 / 18.0 (share / spread90) | 0.367; 0.615; 0.423; 0.368 | — | — | no gain |
| organiser-pinned (b), ceiling | 0.998 / 0.1 / 0.06 | 1.000 / 0.06 | — | — | instrument ceiling |
| **ink effect of tracks, from-scratch fit** (v8-in AUC; densest 4,000-px window / windows holding ≥ 1% of the text, automatic frame) | AUC 0.806 vs 0.802; window 0.514 / 5 vs 0.265 / 16 | AUC **0.809** vs 0.767 (organiser surface: 0.809); window 0.902 / 5 vs 0.486 / 5 | not rendered | — | **pooled AUC 0.802 vs 0.785: +0.017 [+0.007, +0.028]**, closing 34% of the gap to the organiser surface (0.835); contiguity doubles |

All rows [m], single seed. Notes:
- Tracks are the only lever that moves spread90 from 7–16 windings to about 1 on both bands. The isometry and
  normals levers, alone or combined, do not. The tracks warm start began from the converged, hopping 30k fit and
  left it within 3k steps (spread90 10.6 → 1.06).
- This matches the published recipe. The published PHerc0826 First Letters run had tracks on. My G1 baseline
  (tracks off) followed the villa default. On PHerc1447 I built tracks from the published m7 surface prediction.
- Limitations:
  - On the unlabelled w062 band, tracks reduce spread90 from 10–22 to about 3.5 windings, but not to 1. The
    organiser-pinned state itself reaches 2.3 there.
  - The from-scratch tracks fit doubles the winding density at the text [m].
    - Its local fitted ring spacing is 7.7 vox (p10–p90 5.5–11.8), against 13.4 for the tracks-off fit and a papyrus
      pitch of about 14 (13.9 measured label-free from m7).
    - So it counts 3.53 windings between w058 and w060, where the truth is 2.
    - As a result, w058 sits half-way between two fitted windings (v ≈ 63.5) while w060 sits on one (v ≈ 67.06).
    - On the robust bars it passes best-phase share (0.83 / 0.88) and drift range (0.87 / 0.57 windings). It fails
      smoothed switches on w058 (4.65 per 1,000 px) and the between-block count.
  - A global half-ring shift cannot fix it. Shifting the surface by −0.5 ring puts w058 on one winding (window
    0.51 → 0.86) but knocks w060 off (0.90 → 0.62). The phase differs between blocks [m].
  - Late tracks do not fix it [m]. The official schedule switches tracks on late (step 25k of 30k). A fit with the
    track-consistency term held off until step 7,000 of 10k still counts 3.77 windings between w058 and w060. The
    doubling is already present at step 1,000, so timing is not the cause; the pre-registered prediction
    (spacing ≥ 11 vox, count 2.0 ± 0.5) failed.
  - Next step: the track-radius term (weight 50, active from step 0) is the main suspect. A fit with it switched off
    is queued; the fallback is the grad_mag winding-density term.

### 1.5 Does a higher ink AUC mean readable letters? A blinded reading test with catch controls

Every number above is a pixel-level score, but First Letters is judged by people reading letters. I tested the link
directly.

Protocol. I pre-registered the design and the pass rule before building the test kit.
- 89 image crops, each 1.5 × the letter's larger side, in four blocks.
- 14 labelled PHerc1447 letters, each shown four ways:
  - N: automatic surface + v8-in, in the automatic surface's own unrolled frame;
  - B: the same surface mapped into the organiser frame;
  - C: the automatic surface + ink_9um;
  - A: organiser surface + v8-in.
- 33 catch crops, real maps of label-free windows and depth-shuffled inputs. Each was audited in code: 0 labelled-ink
  pixels in the crop or a 128-px margin.
- The reader named the centre letter or answered "no letter", with a confidence of 1–3.
- Pass rule for N: letters named correctly more often than letters named on catch crops (one-sided Fisher p < 0.05),
  with a catch false rate ≤ 15%.

Result (one reader, one session) [m]:

| Condition | Correct letters |
|---|---|
| N | 0 / 14 |
| B | 0 / 14 |
| C | 0 / 14 |
| A | **1 / 14** (confidence 1) |
| Catch crops | 0 / 33 named (no false alarms) |

What this shows:
- Isolated letter crops of v8-in maps are not legible, even on the organiser surface, whose pixel AUC is 0.84.
- The reader was not guessing: 0 false alarms on 33 catch crops.
- Viewed at line scale (about 4,400 × 1,800 px, ink dark), several letters of the same labelled w058 line are
  recognisable (Α, Κ, the bowl of an Ε) [m, unblinded visual check].
- So context and scale carry much of the legibility. This has two practical consequences:
  1. "No letters seen" in letter-sized crops is uninformative.
  2. Reading protocols for unread scrolls should show line-scale strips with matched ink-free nulls, and should first
     pilot the positive control (organiser surface) to confirm that it is readable.

Limitations:
- One reader and 14 letters, so the intervals are wide: A's 1/14 has a 95% CI of 0.01–0.32.
- The block order was fixed (N first), so later blocks are memory-aided.
- My crop display (6-px Gaussian, per-crop 1–99% stretch) is one choice among many.

### 1.6 Can an automatic statistic find text regions in ink maps? A pre-registered negative

Searching a whole unread band costs about 61 GPU-hours of v8-in inference on an 8 GB card (12.5 cm² per GPU-hour)
[m]. That is only worth paying for if something can rank text regions for a human to read.

Test design:
- I pre-registered a family of window statistics:
  - stroke-width consistency, ridge length and ridge-vs-blob ratio;
  - diagonal-stroke share, band-pass kurtosis, text-line periodicity;
  - horizontal ink bands, orientation anisotropy.
- The success bar was fixed before scoring.
- Tuning and testing were swapped between PHerc1447 sheets, so each test sheet was held out.
- Maps: the leave-one-out (`loo`) maps and the base `v8in` maps (from a model that never saw PHerc1447).
- Windows were 1,024 × 2,048 px; CIs come from 2,000 cluster-bootstrap resamples.

Results [m]:
- No finder passed in either round.
  - On held-out w060, TEXT vs unlabelled papyrus scored 0.44 (loo) / 0.43 (v8in).
  - Top-10 precision equalled the random baseline.
- Labelled text is not inkier than unlabelled papyrus at window scale. Ink density gives TEXT vs unlabelled AUC
  0.53 / 0.59 (loo) and 0.25 / 0.19 (v8in). The published pixel AUC (~0.86) is computed only inside the tight
  certainty hull around the labels, so it says nothing about locating text.
- A text-blind detector can pass common nulls.
  - Stroke-width consistency beats phase-randomised surrogates and depth-shuffled-input maps at AUC 1.00 / 0.999.
  - Yet it scores 0.55 for text vs unlabelled, and unlabelled windows beat their own surrogates at 1.00.
  - The depth-shuffled maps differ only trivially: a lower mean and a residual 21-px tile grid.
  - Phase-surrogate verdicts depend on whether the surrogate is drawn before or after smoothing (F2: 1.00 vs about
    0.6).

Consequences:
- A text-region finder needs a text vs real-blank arm, with negatives taken from a densely labelled scroll.
- Depth-shuffled maps should be used only as a "this is a real map" check.
- Surrogates should be built at native resolution.
- Until such a finder exists, a band-wide search for First Letters has no ranking step to pay for.
- Limitation: the "unlabelled" class on PHerc1447 surely contains some unlabelled text, so part of the failure may be
  contamination rather than a missing signal.

### 1.7 End to end on a second known find (PHerc0343)

PHerc0343's First Letters prize was won on 2026-10-08 by Erwin Nieuwlaar. His public repository
`pherc343-first-letters` has his surface, his 18 letter boxes and his `dense_native` model. That gave me a second scroll
with known letters, from a different scan and a different person, on which to test the whole automatic workflow.

Test (the rule was written before any ink was computed):
- Score = mean ink z-score inside each letter box minus a matched null: (I) the gaps between text lines in the same
  rows, (R) a ring around the box. Ink counts as shown only if both 95% CIs (bootstrap over letters) are above 0 and the
  same score on a depth-shuffled render (sabotage) is not.
- Step A, the ink chain alone: his surface, my render, the ink models.
- Step B, everything mine: my tracks-on spiral fit of z 9500–11000 (10k steps, the recipe of §1.4), the official
  flatten, my render, the ink models. His boxes are carried onto my surface through 3D coordinates. At least 9 of the
  18 letters must be covered, or the test counts as not testable.

Results [m]:
- Step A: dense_native +0.77 [0.58, 0.97] (I) / +0.74 [0.62, 0.87] (R), 18 of 18 letters. ink_9um seeds 42 and 43 and
  the mean of the three models also pass. Every sabotage loses (−0.27 to −0.08).
- Step B: my surface covers all 18 boxes. dense_native +0.69 [0.49, 0.89] (I) / +0.71 [0.51, 0.92] (R), all 18 letters
  positive on both. Mean of the three models (ink_9um s42, s43, dense_native): +0.56 [0.42, 0.70] / +0.56 [0.40, 0.73].
  Sabotage: −0.39 / −0.25 and −0.35 / −0.23.
- At line scale the ink on my surface forms letter shapes inside the boxes. I looked at this knowing where the letters
  were, so it is not a blind reading.
- Ranking control: the tile score used to pick regions for a closer look (p95 − p50 of the depth-mean dense_native map
  in 512-px tiles) puts 7 of the 15 letter tiles in the top 10% of 111 tiles (my pre-registered bar was 50%, so it
  fails narrowly). Letter tiles vs other tiles: AUC 0.89, binomial p = 0.0003. On the sabotage maps: 1 of 15, AUC 0.38.

What it shows and what it does not:
- The automatic workflow puts ink where real letters are, on its own surface, against matched nulls, and a sabotage
  loses. That is the precondition for running it on unread scrolls.
- It is a reproduction on a scroll that was already read. The scores are measured where the letters are known to be,
  so this is not a test of finding text blind (see §1.6).
- Software differed slightly from the PHerc1447 runs: torch 2.14.1 on the rented GPU, and the fused Triton path for the
  fit's inverse transform, which the spiral-fit code provides. Without it, a 100 M-point track band ran out of memory
  on a 24 GB card.

## 2. Stage-loss decomposition (G1/G2 with ink_9um; G3 with v8-in)

| Stage | Contrast (identical pixels) | ΔAUC [95% CI] |
|---|---|---|
| Ink model | v8-in vs ink_9um, organiser surface (0.836 vs 0.707) | **+0.129** [m] |
| Ink model on the automatic surface | v8-in vs ink_9um (0.785 vs 0.665) | +0.12 [m] (band 2: +0.18 [m, from the G3 write-up]) |
| Surface, ink_9um | automatic 30k flattened vs organiser surface, same renderer (0.665 vs 0.707) | **−0.042 [−0.068, −0.015]** [m] |
| Surface, v8-in | automatic + geometric order + 5-phase average vs the released organiser map (0.790 vs 0.836) | **−0.046 [−0.061, −0.032]** [m] |
| Surface, v8-in, held-out band 2 | automatic vs the released map, single phase (0.746 vs 0.812) | **−0.066 [−0.089, −0.048]** [m] |
| Depth-order decision | label-free separation rule vs geometric rule, 7 automatic arms | 0 to **−0.121** pooled; −0.175 on one winding [m] |
| Flatten | flattened vs raw, at 5k / 10k / 30k steps | +0.006 / −0.004 / +0.013, i.e. ≈ 0 [m] |
| Render | my port vs the organisers' render, same surface and canvas | −0.002 [m] |
| Tiling phase, ink_9um | 5 canvas offsets | SD 0.010 (range 0.026); averaging the 5 phases gives +0.015 [m] |
| Tiling phase, v8-in | 5 canvas offsets | SD 0.0014; averaging the 5 phases gives +0.0047 [+0.0022, +0.0076] [m] |

Why the surface loses, at the pixel level [m]:
- Position scatter does not shrink with training. The automatic sheet scatters around the true sheet with an IQR of
  −3.7…+3.1 vox (w058) at 30k steps, about the same as at 1k (−3.4…+3.5). Only the tilt improves: median
  14.9° → 6.1°.
- ink_9um tolerates offsets on one side only. It is flat from −4 to 0 vox (0.705–0.707), then loses 0.07 at +4 and
  0.13 at +6.
- v8-in loses at every offset. By local offset (observational), it loses only 0.009 where the automatic surface is
  within ±1 vox of the organiser surface, about 0.03 at 1–4 vox, 0.05 at 4–6 vox and 0.07–0.10 beyond.
- Conclusion: under v8-in, surface position is essentially the whole remaining AUC gap. For reading, winding
  consistency (§1) is the bigger problem.

## 3. Depth order: replace the separation heuristic with geometry

Official `--direction both` plus the separation rule (mean p above 0.5 minus mean p at or below 0.5), on 7 automatic
arms (4 checkpoints × flattened/raw; it1000 raw not run):

| Arm | Wrong on | Fixed-order AUC | Separation-rule AUC | Cost |
|---|---|---|---|---|
| it1000 flat | w058 | 0.634 | 0.538 | −0.096 |
| it5000 flat | w060 | 0.644 | 0.637 | −0.008 |
| it5000 raw | w058 | 0.638 | 0.517 | **−0.121** (w058 alone: 0.685 → 0.510) |
| it10000 raw | w060 | 0.662 | 0.637 | −0.025 |
| it30000 raw | both | 0.653 | 0.552 | −0.101 |
| it10000 flat, it30000 flat | — | 0.658, 0.665 | same | 0 |

All rows [m].

- Geometric rule: at matched points, N_auto · N_org > 0 on 100.00% of 2.9M (w058) and 1.2M (w060) pixels. So
  "flip-normals render, read forward" is physically the order the organisers' README prescribes for their own
  renders [m].
- Order check under v8-in: the reverse order gives 0.4965 (chance) and forward gives 0.785 [m].
- Separate case: on PHerc0343P the separation rule also picks the wrong order for v8-in, 0.437 vs 0.807 [m].
- Caveat: I validated this on PHerc1447 only. A scroll with the opposite catalogue z-direction must be checked.

## 4. The public PHerc0343P labels: mirrored and offset

The data is `scrollprize/datasets` `ink/unused/P343p`, segment `20250511003658`. Its labels sit on a 2.215 µm render
grid.

Registration [m]:
- Orientation search (8 flips and transposes at about 35 µm): only the left–right flip matches (NCC 0.207 vs at most
  0.062 for the other seven).
- Global affine rejected: the best affine leaves 1.87 px RMS on 9 tiles (bar 1.5 px), because the two renders differ
  by a non-affine warp.
- Local placement: instead, I placed the labels by local template matching, with shift (0, −51) px at nominal scale,
  peak NCC 0.20, peak/median 8.6.
- No negatives: the set's "supervision" mask is identical to its ink labels.

ink_9um seed42, full canvas, forward order [m]:

| Label placement | Whole valid canvas | Labels' bbox + 160 px |
|---|---|---|
| PR-29's placement (scale only, origin 0, no flip) | **0.406** (PR-29 reported 0.418) | 0.430 |
| flipped only, no shift | 0.557 | 0.557 |
| **registered (flip + measured shift)** | **0.696** | **0.723** |

- PR-29's below-chance "held-out control fails" result is a label-placement artefact. ink_9um *does* show ink there.
- Conclusions built on that control should be re-checked.
- There is a smaller registration issue on 0009B: a ±8 px search window was clipped, and the true offset is (11, 9).
  It costs at most 0.001 AUC [m].

## 5. Released v8-in stitching: normalise by blend weight, not tile count

`ink8um/inference.py` (file sha256 `bca4a2ba…f38` [m]) computes `predict_stack = Σ(p·g) / count` instead of Σ(p·g) / Σg.
So every released map equals the correct map times W = Σg/count, a field that depends only on the tile layout.

Size of the error:
- Interior W ranges from 0.611 to 0.758–0.891 depending on the canvas: a ±10–19% grid with period 21 px [m, G3
  write-up].
- At coverage edges W reaches 0.38–0.94.

Paired AUC on the 4.16M px:

| Map | Correction gain | Sabotage (W shifted by 10 px) |
|---|---|---|
| released map | **+0.0008 [+0.0007, +0.0009]** | −0.0006 [−0.0011, −0.0002] |
| my run on the organiser surface | +0.0008 [+0.0006, +0.0009] | −0.0009 |
| automatic surface | +0.0002 [−0.0003, +0.0007] | −0.0007 [−0.0015, +0.0001] |

All rows [m]. I re-ran the automatic-surface row from this package's code and it matched exactly.

- The case for fixing it is exactness and appearance, not AUC: a per-crop contrast stretch turns the grid into
  visible lines on faint ink.
- Patch: `patches/v8in-stitching-normalise-by-blend-weight.patch` (`count += blend`, plus the docstring). It applies
  cleanly to the released file [m].
- Already-released maps can be corrected without re-running the model by dividing by W (`g3_score.wfield_released`).

## 6. Tools (in `repo/`, MIT)

| Tool | What it does | Validation [m] |
|---|---|---|
| `g12lib.py` | `vc_render_tifxyz` linear-path port: bilinear positions, per-vertex normals, trilinear layers. Plus a tifxyz reader and an on-demand CT chunk mirror | vs the organisers' w058 render (256² tile × 24 layers): r = 0.99987 (min 0.99984), 100% within 1 grey level, 50% exact. The wrong pixel-centre conventions score 0.989 / 0.878, and flipped normals 0.11 (G1/G2 write-up) |
| `winding_metric.py` | Text fragmentation of any tifxyz surface vs reference surfaces | Self-test: identity 1.000 / 1 stretch; shuffled-block sabotage 0.670 / 2 |
| `surf_wind.py` | Winding consistency read straight from a fit checkpoint (no flatten, no render), 14 s on CPU | Reproduces 0.280 / 14 and 0.473 / 4 (rerun from this package) |
| `surf_eval.py` | The fit's own loss terms at fixed checkpoints, paired with common random numbers | Reproduces the logged loss (2.54 vs 2.5); paired SE 0.02–0.03 |
| `g12_g2.py`, `g3_score.py` | Stage-loss scorers with built-in sabotage arms (±10 vox, a neighbouring sheet) and an identity check | Identity mapping = direct score to the last digit; sabotage → 0% adherence, AUC 0.49–0.55 on band 1 (0.56 / 0.64 on band 2, against 0.81) |
| `g12_umbilicus.py` | Automatic umbilicus (port of VC3D `align_and_extract_umbilicus`) | Seed spread 0.5–2.8 mm, within the tool's documented "few mm" |
| `vlib/metrics.py` | Exact AUC, block bootstrap, C2 null, `eval_forme` elongation | Self-test passes (Mann–Whitney agreement, null 0.499) |
| `ensemble/speed/fastvilla.py` | Faster villa ink inference: the patch filter, robust-MAD normalisation and overlap-add move from a CPU worker pool to the GPU | Bit-identical maps to the default path on 57 M px (three workloads, 16 and 6 CPUs) on an RTX 4060, and on 236 M px on an RTX 4090. Unit test on real and edge-case patches: 100% of voxels; its sabotage arm (divisor moved by one float32 ulp) fails. 3.1× faster on the 4060 (3.63 → 1.18 s per valid Mpx per depth), 2.4–2.8× on the 4090 |

### 6.1 Why villa inference was slow, and the fix

On the default path about 70% of the wall time was CPU work: villa's robust-MAD normalisation of each 64³ patch runs
in a 6-worker process pool, and the serial `pool.map` per batch of 16 patches (278 KB in, 1.1 MB out per patch) is the
bottleneck. More CPUs do not help (6 and 16 CPUs give the same speed; 14 workers is slower), and the GPU sits at
27–34% use. The forward pass itself is compute-bound (2.70 ms per patch at batch 16; batch size, CUDA graphs,
`cudnn.benchmark` and channels-last change it by 0.97–1.05×).

The fast path sends the tile's uint8 window to the GPU once and does the zero-patch filter (integral image), the patch
slicing, the normalisation and the Hann overlap-add there. The normalisation reproduces villa's `normalize_robust`
exactly: 256-bin per-patch histograms give the order statistics, the 1st and 99th percentiles come from `np.percentile`
itself (run on a small constructed array with the same length and the same order statistics around the interpolation
point, cached), and clip, median, MAD and the division run in float32. Patches that take numpy's MAD fallback branch
(12–16% here, mostly ones over half zero-padding) get their standard deviation from the CPU. After the change the path
is forward-bound (forward 74% of the time). It is opt-in (`depth_sweep_fast.py --villa-fast`); a numpy or torch update
should be re-checked with the unit test.

A correction. My first version computed the percentiles with its own float64 interpolation. An independent
re-implementation showed that this differs from numpy by one float32 step at the clip bounds for 14% (1st percentile)
and 7.7% (99th) of value pairs at n = 278,528. The maps were still bit-identical in every comparison, because the
median subtraction absorbed the difference, so the original unit test could not see it. The unit test now checks the
percentiles directly: the old step fails it (210 and 62 mismatches in 1,500 cases), the current one passes (0 and 0),
on CPU and GPU. With the fix the normalised tensors are bit-exact on 609 real and edge-case patches, the maps are
still bit-identical, and the speed is unchanged [m].

## 7. An automated screening pipeline for one rented GPU (`tools/cloud/`)

The kit runs the whole First Letters workflow on one rented GPU, one 1,000-slice band at a time: fit inputs (lasagna
normals and surface tracks) → tracks-on spiral fit → official flatten → render → a whole-band dense_native screen at
d −2/0/+2 → a tile ranking → the three-model ensemble at d −6..+6 on the top tiles → line-scale strips and crops for a
human reader. Every step is resumable, a cost guard runs before every step and every strip, and the job stops itself
at a budget. A mocked dry run of all jobs passes 40/40 checks on a laptop.

Validation:
- On PHerc0343 the same chain, on its own surface, puts ink inside all 18 of the winner's letter boxes (§1.7).
- The ranking step was checked on the same window (§1.7): letter tiles vs other tiles AUC 0.89, 7 of 15 letter tiles
  in the top 10%.
- Tiles near the edge of the valid surface produced bright bars that took ranking slots. Scoring only pixels at least
  128 px inside the mask removed most of them and left the PHerc0343 ranking unchanged (AUC 0.88, 6 of 14) [m].

Measured cost [m]. I ran it on six 1,000-slice bands of unread scrolls on a rented RTX 4090 at $0.91 per hour. This
report makes no claim about their text.

| Step | Time per band (steps timed on the first two bands) |
|---|---|
| Fit inputs (normals + tracks) | 11–17 min |
| Spiral fit, 10k steps | 7–12 min |
| Flatten | 1–2 min |
| Whole-band screen (about 1,030–1,195 valid Mpx) | 21–26 min |
| Ensemble on the top 10% of tiles | 37–43 min |
| **Whole band (all six bands)** | **1 h 23 min to 2 h 20 min, $1.25–2.10** |

In all, about 5,000 cm² of surface were screened for about $10.60, roughly $0.002 per cm².

Problems I hit on the first real pod, now fixed in the kit:
- The fit ran out of memory on a 24 GB card with a 100 M-point track band. The non-fused inverse transform builds a
  points × windings tensor; the spiral-fit code's fused Triton path (`FIT_SPIRAL_TRITON=1`) used 6 GB.
- A second lock file pulled a newer torch into one environment through a dependency, so the pod did not match the
  tested versions. The environments now install their pinned files with `--no-deps`.
- The flatten step needs villa's `lasagna/` directory, which a sparse checkout of `vesuvius` and `spiral-fitting`
  leaves out.
- The villa models spent most of their time in CPU normalisation; the fast path (§6.1) cut the screen to about
  0.4 s per valid Mpx per depth on the 4090.

## 8. Is the surface on the sheet? A check, and what snapping buys (`tools/qa/sheet_offset.py`)

The First Letters workflow guide asks people to confirm that the mesh stays on one physical sheet. I wanted a number
for that. `sheet_offset.py` samples the organisers' m7 surface prediction (a binary sheet mask) along each vertex
normal, from -15 to +15 voxels, and takes the centre of the nearest sheet. This is the measurement inside the
PHerc0343 winner's `snap_meshes.py`, run on a sample of 128^3 blocks streamed from the public bucket. Nothing is
stored.

Controls first [m]:
- The winner's snapped meshes: 70% and 63% of vertices within 2 voxels of a sheet centre; median offset 1.0 and 1.5.
- The same mesh pushed 5 voxels along its normal: 9% within 2 voxels, median 5.0, signed median -4.75. The tool
  recovers the push.

My surfaces [m]: our PHerc0343 surface, and the six unread bands screened in section 7, all sit a median 3.5-5.0
voxels off the sheet. Only 26-35% of vertices are within 2 voxels, and 50-59% are 4 or more voxels off. That is close
to random placement. The winner snaps his fitted windings onto the sheet prediction before flattening; I never did.

What snapping buys, on known letters [m]. I snapped our PHerc0343 surface with his script (defaults, level 0) around
his letters. 95% of vertices found a sheet, with a median move of 3.0 voxels. I then re-rendered three windows with
the same models and scored each letter box against its own interline boxes and ring (the section 1.7 rules). The
rule was fixed before scoring: snapping counts as a gain if the mean per-letter contrast rises in at least 2 of the
3 windows and over all 18 letters.

| Depths | Unsnapped (mean over 18 letters) | Snapped | Per-letter change [95% CI] | Letters improved |
|---|---|---|---|---|
| d -2/0/+2 (the screen) | 0.582 | 0.611 | +0.029 [-0.017, 0.077] | 10 of 18 |
| d -4..+4 | 0.337 | 0.404 | +0.069 [0.035, 0.103] | 15 of 18 |

The rule passes (3 of 3 windows), but the gain at the screen's own depths is within noise. Snapping mainly keeps the
signal alive away from d = 0 (d +4: 0.10 to 0.27 and 0.24 to 0.48 in two windows). So an off-sheet surface costs less
than I expected with 66-layer models, and it does not explain why the six bands in section 7 showed no text.

The kit now has an opt-in `SNAP=1` step between flatten and plan (`tools/cloud/py/snap_meshes_nieuwlaar.py`, his MIT
script with its licence; `py/m7path.py` finds the prediction). If snapping fails, the job screens the unsnapped mesh
and says so. Pair it with a wider stage-2 depth set.

## 9. A model-free search for lead-based ink in all 21 eligible scrolls (`tools/qa/lead_screen.py`)

Seiler et al. (PLOS ONE 21(9) e0353485, September 2026) showed on a burned model scroll that lead in ink absorbs up
to about 25 times more X-rays than carbonized papyrus, and is visible at 25 ug/cm2. They note that some Herculaneum
papyri contain lead and that nobody has searched the collection. If a scroll had lead ink, its letters should be
bright on the sheet surface in the raw CT, with no ink model needed.

`lead_screen.py` takes random 128^3 CT blocks on papyrus (chosen from the m7 sheet mask) and measures three things:
how strongly the brightest voxels (above the scroll's 99.9th percentile) concentrate near sheets; how many thin,
sheet-like bright components sit on the sheets; and saturation near sheets.

Positive control [m]: on PHerc0343 I added +100 grey levels to a striped one-voxel layer on the sheet surface. The
enrichment went from 1.20 to 2.03, thin bright components from 0.76 to 4.08 per 10^6 sheet voxels, and saturation
rose 15 times. The screen can see a strong lead-like layer.

Result [m], 24 blocks per scroll: thin bright components 0.6-2.2 per 10^6 sheet voxels in every scroll, including
two known baselines, except two eligible scrolls at 4.75 and 3.16. On 60 fresh blocks those two gave 0.71 and 1.50,
against 1.99 and 0.89 for two scrolls from the same scanner settings. They did not replicate. Face-on views of their
largest bright components show isolated 5-10 voxel specks (inclusions) on ordinary papyrus, not strokes.

Verdict: no eligible scroll shows a strong lead-ink signature. How weak a layer the screen can see [m]: injecting
weaker layers (about 5% of sheet-surface voxels, 24 blocks) moved the enrichment on an 8.64 um scan from 1.20 to 1.17
(+25 grey levels), 1.51 (+50) and 1.77 (+75), and on a 9.362 um scan from 0.92 to 0.95, 1.01 and 1.31. The highest
value among the eligible scrolls was 1.24. So a layer of about +50 grey levels (8.64 um scans) or +75 (9.362 um scans)
over 5% of the surface would have stood out; weaker or sparser lead is not excluded, and I cannot convert grey levels
into ug/cm2 for these scans. Each scroll was sampled at about 26 mm^3 of sheet-rich CT. Per-scroll numbers (unread scrolls unnamed) are in `results/qa/lead_screen.json`.

## Checks that could have failed

- Renderer: the known-answer test fails for the wrong conventions [m].
- ±10-vox shifted sheets: both lose, with AUC 0.491 / 0.545 and adherence 0% [m].
- Neighbouring organiser sheet as the surface: AUC 0.500 [m]. I first labelled this sheet "adjacent", but it is
  actually 2 windings away, because w058, w060 and w062 are every second winding.
- C2 block-derangement nulls: 0.49–0.51 on every arm [m].
- Reverse depth order: 0.4965 [m].
- Stitching correction with a misaligned W: loses [m].
- Winding-metric shuffle sabotage: loses [m].
- Sheet-offset tool: a mesh pushed 5 voxels off the sheet scores 9% on-sheet (vs 70%) and the push is recovered [m].
- Lead screen: an injected +100 layer raises every statistic, and +50 still raises the enrichment on an 8.64 um scan
  (+75 on a 9.362 um scan), though not the thin-component count; the two highest scrolls did not replicate [m].
- A raw-CT intensity profile along the normal could not tell a verified surface from one shifted by half a sheet
  spacing, so I dropped it and used the sheet prediction instead [m].
- A position snap that looked like a gain: on adherence, its own random-sign sabotage arm also gained, which voided
  it [m].

## Limitations

- Scope: one scroll (PHerc1447, 8.64 µm), two organiser windings and two bands; band 2 has only w060. I used one
  ink_9um checkpoint for the decomposition, and only the v8-in base model.
- Tracks: tracks were off, because none are published for PHerc1447. The published PHerc0826 First Letters run had
  tracks on (villa `6847063`, where `input_use_tracks` still defaulted to true). The hop rate with tracks on is one
  of the pending §1.4 levers.
- Umbilicus: automatic, not human-annotated. I checked it visually on one slice.
- Renderer: my validated Python port, not the VC3D binary.
- Fit path: eager CUDA, not the fused Triton path (whose JIT did not finish under WSL); the mathematics is the same.
- Uncertainty: one fit per band, so the CIs and SEs exclude fit-to-fit (seed) variance. The block-bootstrap CIs also
  exclude tiling-phase variance (ink_9um SD 0.010).
- Elongation (letter shape): only 8 letters qualified, so these results are descriptive only.
- Labels in the diagnostic: the Step 1 (b) state uses organiser labels as patches. It is a diagnostic, not a method.
- Not verified here: I did not re-fetch PR-29's 0.418 or the PHerc0826 recipe details for this package.

## Reproduction (details and data layout in `repo/README.md`)

```
pip install -r requirements.txt
```

```
export VC_ROOT=$HOME/vesuvius
```

```
python tools/vlib/metrics.py
```

```
python tools/winding_metric.py --selftest
```

```
python tools/g12_validate_render.py 2950 3900 256
```

```
python tools/winding_metric.py $VC_ROOT/work/g1g2/flat/it30000.tifxyz --ref w058 --ref w060 --zband 13150 14050 --mask labels --out wm_band1.json
```

```
python tools/surf_wind.py FIT.ckpt $VC_ROOT/data/spiral/PHerc1447/umbilicus.json $VC_ROOT/work/g1g2/org/w058,$VC_ROOT/work/g1g2/org/w060 wind.json --stride 2
```

The full chain (fit → flatten → render → infer → score) and the v8-in, 0343P and stitching runs are in the README.

## Credits, data terms, licence

- Code I built on:
  - ScrollPrize/villa: `fit_spiral.py`, `flatten_spiral_checkpoint.py` and the ink `infer` tool.
  - VC3D's `vc_render_tifxyz` and `align_and_extract_umbilicus`, whose behaviour I re-implemented in Python.
- Data and models:
  - The Vesuvius Challenge organisers: data, surfaces, labels, lasagna normals, m7 predictions, ink_9um.
  - YoussefMoNader: v8-in and the PHerc1447 surfaces dataset.
- Methods I reused or reproduced:
  - AndreasHad04 (villa #1898, `villa-apple-silicon` bench): the AUC definition, the C2 null and the label transfer.
  - liliandevarieux (`vesuvius-841`): the `eval_forme` elongation metric and the pre-registered benchmarks, including
    the PR-29 control I re-registered.
  - millerandmuller: the published end-to-end PHerc0826 First Letters run.
- Data terms:
  - Data: Vesuvius Challenge open data, CC-BY-NC 4.0. The repository contains no scroll data, only summary numbers.
  - Legacy-data citation: EduceLab-Scrolls (Parsons et al., 2023, arXiv:2304.02084).
  - All measurements are on publicly labelled text. No output from an unread scroll is included.
- Licence: MIT.
