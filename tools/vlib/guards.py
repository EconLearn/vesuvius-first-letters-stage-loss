"""Run guards: data-terms gate, GPU courtesy check (for shared machines), disk guard, leakage tripwire."""
from __future__ import annotations

import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

import vcpaths as _P

ROOT = Path(_P.ROOT)
TERMS = ROOT / "TERMS_ACCEPTED"


def require_terms():
    """Scripts that read scroll data refuse to run until you have accepted the Vesuvius Challenge data terms
    yourself (Data Agreement form + `vesuvius.accept_terms --yes`) and created $VC_ROOT/TERMS_ACCEPTED."""
    if not TERMS.exists():
        sys.stderr.write(
            f"REFUSING TO RUN: {TERMS} does not exist.\n"
            "Accept the Vesuvius Challenge data terms yourself (Data Agreement form + "
            "`vesuvius.accept_terms --yes`), then create that file.\n")
        sys.exit(2)


def renice():
    try:
        os.nice(19 - os.nice(0))
    except Exception:
        pass
    try:
        subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], check=False, capture_output=True)
    except Exception:
        pass


def free_gb():
    return shutil.disk_usage(_P.disk_guard_path()).free / 1e9


def disk_guard(need_bytes=0, min_free_gb=None):
    min_free_gb = _P.MIN_FREE_GB if min_free_gb is None else min_free_gb
    free = free_gb()
    if free - need_bytes / 1e9 < min_free_gb:
        raise SystemExit(f"DISK GUARD: {_P.disk_guard_path()} has {free:.1f} GB free; this step needs "
                         f"{need_bytes/1e9:.1f} GB and must leave >= {min_free_gb:.0f} GB. Stopping.")
    return free


def gpu_status():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, total, util = (float(x) for x in q.stdout.strip().split(",")[:3])
    return dict(used_mb=used, total_mb=total, util=util)


def gpu_check(need_mb, wait_s=0, max_util=60):
    """Wait until need_mb of GPU memory is free and utilisation is low (useful on a shared GPU).
    Set VC_SKIP_GPU_CHECK=1 to skip."""
    if os.environ.get("VC_SKIP_GPU_CHECK") == "1":
        return {}
    t0 = time.time()
    while True:
        s = gpu_status()
        if s["total_mb"] - s["used_mb"] >= need_mb and s["util"] <= max_util:
            print(f"[gpu_check] ok: {s}", flush=True)
            return s
        if time.time() - t0 >= wait_s:
            raise SystemExit(f"GPU BUSY: {s}; need {need_mb} MB free and util <= {max_util}%. "
                             "Not starting. Re-run later, or set VC_SKIP_GPU_CHECK=1.")
        time.sleep(30)


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tree_hashes(paths):
    """sha256 of every file under the given files/dirs (zarr chunks included)."""
    out = {}
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in files:
            out[str(f)] = sha256_file(f)
    return out


def leakage_tripwire(train_inputs, heldout_inputs, name="run", record_dir=None):
    """Raise if any held-out file (by content hash) or path is among the training/selection inputs."""
    record_dir = Path(_P.OUT) / "tripwire" if record_dir is None else Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    tr, ho = tree_hashes(train_inputs), tree_hashes(heldout_inputs)
    shared = set(tr.values()) & set(ho.values())
    # zarr metadata files (.zarray etc.) and all-empty chunks can be byte-identical across arrays: ignore
    # hashes of files < 1 KiB, which carry no label content.
    small = {h for p, h in {**tr, **ho}.items() if Path(p).stat().st_size < 1024}
    shared -= small
    rec = dict(name=name, n_train_files=len(tr), n_heldout_files=len(ho), shared_hashes=len(shared),
               when=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (record_dir / f"{name}.json").write_text(json.dumps(rec, indent=1))
    if shared:
        raise SystemExit(f"LEAKAGE TRIPWIRE: {len(shared)} held-out file(s) appear in the training inputs of {name}")
    return rec
