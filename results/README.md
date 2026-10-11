# Results behind FINDINGS.md

These files contain summary numbers only: AUCs, CIs, shares and loss terms. They hold no scroll data. Absolute
paths are shown as `$VC_ROOT/...`.

| FINDINGS item | File(s) |
|---|---|
| Stage-loss table, ink_9um arms, learning curve, depth-order costs | `g1g2/auto_table.json`, `g1g2/g2_scores_it30000.json` |
| Tiling-phase floor (ink_9um) | `g1g2/phase_floor.json` |
| Offset dose-response | `g1g2/dose_response.json` |
| Geometry: offsets, tilt | `g1g2/geom_it30000flat_it30000raw.json` |
| Offset-only model | `g1g2/offset_model_it30000flat.json` |
| Renderer known-answer test | `g1g2/render_validation.json` |
| v8-in arms, band 1 (5 phases + average) | `g3/g3_scores_b1_B1.json` |
| v8-in arms, band 1, maps divided by W | `g3/g3_scores_b1_flat_B1.json` |
| Reverse-order check | `g3/g3_scores_b1_B1rev.json` |
| v8-in arms, band 2 (held out) | `g3/g3_scores_b2_B1b2.json` |
| Winding metric, band 1 (cap_switch 3) | `g3/wm_band_3.json` |
| Winding metric, band 2 | `g3/winding_b2_labels.json` |
| Pixel-level fragmentation check | `g3/auto_winding_consistency.json` |
| Position-snap test and its random-sign sabotage | `g3/snap_adh_it30000snap.json`, `g3/snap_adh_it30000snapSR.json` |
| Stitching, paired test | `g3/stitching_paired_original_run.txt` (that run printed three lines and then stopped before writing JSON; the B1 line was re-run from this repository's code and matched to all printed digits) |
| Fit-loss diagnosis, Step 1 | `surf/eval_step1a.json`, `surf/eval_step1b.json` (normals residual uncapped), `surf/eval_step1b_rob10.json` (capped at 10°); paired differences in `paired_minus_first` |
| Winding share during the (a+), (b) and (d) runs | `surf/watch_a_plus.jsonl`, `surf/watch_b_follow.jsonl`, `surf/watch_d_release.jsonl` |
| Checkpoint-native winding metric along the official learning curve | `surf/wind_curve_it*.json`, `surf/wind_it30000_s2.json` |
| Fitted winding density | `surf/density_it30000.json` |
| PHerc0343P registration and the PR-29 placement comparison | `e1/0343P_mapping_report.json`, `e1/0343P_mapping_check.json` |
| E1 scale test (all arms) | `e1/scores_final.json` |
