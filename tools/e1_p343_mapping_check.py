"""E1 instrument check on PHerc0343P (label placement).

ink_9um seed42 (official villa infer, in-process, both orders) on the FULL 8.64 um canvas of segment
20250511003658, then AUC against three label placements, each inside (a) the whole valid canvas and (b) the
labels' own bbox + 160 px:
  registered  = E1's placement (fliplr + measured local shift (0, -51) at nominal scale 8.64/8.86)
  pr29        = liliandevarieux PR-29's placement: scale 8.64/2.215, origin 0, no flip (assumed, not measured)
  flip_only   = fliplr, nominal scale, no shift
If pr29 scores near/below 0.5 while registered scores well above, PR-29's 0.418 was a placement artefact.
"""
import vcpaths as _P  # noqa: E402  (path configuration, see vcpaths.py)
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, tifffile, zarr

sys.path.insert(0, _P.TOOLS)
from vlib import vio, metrics as M
import e1_scale_test as E


def main():
    OUTD = Path(f"{_P.WORK}/e1/p343_full"); OUTD.mkdir(parents=True, exist_ok=True)
    VOL = vio.s3_local("PHerc0343P/segments/20250511003658-tifxyz/surface-volumes/8.64um-1.2m-116keV-volume-20250521134555.zarr")
    out_f = OUTD / "s42_full.tif"
    if not (OUTD / "s42_full_reverse.tif").exists():
        import multiprocessing as mp, logging, torch
        mp.set_start_method("fork", force=True)
        from vesuvius.ink_detection.inference import infer as INF
        logging.basicConfig(level=logging.INFO)
        args = INF.normalize_inference_paths(INF.parse_args([VOL, str(E.CKPT["s42"]), str(out_f), "--direction", "both",
                                                              "--batch-size", "32", "--num-workers", "4", "--no-compile"]))
        E.wait_gpu(2500)
        cfg = INF.configure_model(args)
        model, dev = INF.prepare_model_for_inference(cfg.model, gpu_ids=args.gpu_ids, compile_model=False, compile_mode=args.compile_mode)
        cfg = replace(cfg, model=model)
        for d in ("forward", "reverse"):
            INF.infer_single_zarr(args=args, input_zarr=VOL, configured_model=cfg, device=dev,
                                  output_tiff=INF.resolve_single_output_path(out_f, direction=d, requested_direction="both"), layer_direction=d)

    a8 = vio.open_level(VOL, 0); H, W = a8.shape[1:]
    valid = np.asarray(a8[15]) > 0
    preds = {d: tifffile.imread(OUTD / ("s42_full.tif" if d == "forward" else "s42_full_reverse.tif")).astype(np.float32) / 255.0
             for d in ("forward", "reverse")}
    rep = json.loads((E.WORK / "labels" / "0343P_mapping_report.json").read_text())["20250511003658"]
    i0, i1, j0, j1 = rep["crop"]
    z = np.load(E.WORK / "labels" / "0343P_20250511003658.npz")
    reg = np.zeros((H, W), bool); reg[i0:i1, j0:j1] = z["label"]
    b = f"{_P.DATA}/hf_bucket/scrollprize_datasets/ink/unused/P343p/20250511003658/20250511003658"
    ink = np.asarray(zarr.open_array(b + "_inklabels.zarr/2", mode="r")[:]).max(0) > 127
    S = 8.64 / (2.215 * 4)
    pr29 = vio.resample(ink.astype(np.uint8), 0, H, 0, W, 0, 0, S, fill=0)[0].astype(bool)
    flip = vio.resample(np.ascontiguousarray(ink[:, ::-1]).astype(np.uint8), 0, H, 0, W, 0, 0, S, fill=0)[0].astype(bool)
    out = dict(sep={d: M.separation(p, valid) for d, p in preds.items()})
    for lname, L in (("registered", reg), ("pr29_origin0_noflip", pr29), ("flip_only_noshift", flip)):
        y0, y1, x0, x1 = E.bbox(L, 160, (H, W)); box = np.zeros((H, W), bool); box[y0:y1, x0:x1] = True
        for sname, Sm in (("whole_valid_canvas", valid), ("labels_bbox160", box & valid)):
            for d, p in preds.items():
                out[f"{lname}|{sname}|{d}"] = dict(auc=round(M.auc(p, L & Sm, Sm), 4), label_px=int((L & Sm).sum()), support_px=int(Sm.sum()))
    print(json.dumps(out, indent=1))
    (E.WORK / "labels" / "0343P_mapping_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
