#!/usr/bin/env python3
"""
Multi-dataset experiment suite runner.

Orchestrates the full autoresearch experiment loop across multiple datasets,
keeping each dataset's results cleanly isolated.

Usage:
    # Run full suite (convert -> tokenize -> agent run for each dataset)
    python run_suite.py

    # Run a single dataset
    python run_suite.py --dataset fineweb-edu

    # List available datasets and their status
    python run_suite.py --status

    # Skip datasets that already have results
    python run_suite.py --skip-completed

    # Customize per-dataset experiment count
    python run_suite.py --max-experiments 80

Each dataset gets:
  - Its own data + tokenizer profile in ~/.cache/autoresearch/profile_<name>/
  - Its own results file in results/<name>/results.tsv
  - Its own git branch: autoresearch/<tag>-<name>
"""

import os
import sys
import json
import hashlib
import shutil
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
RESULTS_DIR = PROJECT_ROOT / "results"
PROFILES_DIR = CACHE_DIR / "profiles"

# Dataset run order (priority order from plan)
DATASET_ORDER = [
    "climbmix",
    "fineweb-edu",
    "cosmopedia-v2",
    "slimpajama",
    "fineweb-edu-high",
    # --- Round 2 ---
    "fineweb",
    "github-code-python",
    "pubmed-abstract",
]

# ---------------------------------------------------------------------------
# Content fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint_data_dir(data_dir):
    """Generate a content fingerprint from the first shard's first document."""
    shard_path = data_dir / "shard_00000.parquet"
    if not shard_path.exists():
        return None

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(shard_path))
        rg = pf.read_row_group(0)
        first_doc = rg.column("text").to_pylist()[0]
        doc_hash = hashlib.sha256(first_doc.encode("utf-8")).hexdigest()[:16]
        sample = first_doc[:200].replace("\n", " ")
        return {"hash": doc_hash, "sample": sample}
    except Exception as e:
        print(f"  WARNING: Could not fingerprint data: {e}")
        return None


def _validate_fingerprint(profile_dir, expected_name):
    """Check if a profile's fingerprint matches its stored identity."""
    meta_path = profile_dir / "meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    stored_fp = meta.get("fingerprint")
    if not stored_fp:
        print(f"  WARNING: Profile '{expected_name}' has no fingerprint -- cannot validate")
        return False

    data_dir = profile_dir / "data"
    current_fp = _fingerprint_data_dir(data_dir)
    if not current_fp:
        return False

    if current_fp["hash"] != stored_fp["hash"]:
        print(f"  ERROR: Profile '{expected_name}' fingerprint mismatch!")
        print(f"    Stored:  {stored_fp['sample'][:80]}...")
        print(f"    Actual:  {current_fp['sample'][:80]}...")
        return False

    return True


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

def save_profile(name, force=False):
    """Save current data + tokenizer as a named profile with fingerprint."""
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists() and not force:
        print(f"  Profile '{name}' already exists, skipping save (use --rebuild-profiles to force)")
        return

    if profile_dir.exists() and force:
        print(f"  Removing stale profile '{name}'...")
        shutil.rmtree(profile_dir)

    profile_dir.mkdir(parents=True, exist_ok=True)

    if DATA_DIR.exists():
        data_dest = profile_dir / "data"
        print(f"  Saving data -> {data_dest}")
        shutil.copytree(DATA_DIR, data_dest)

    if TOKENIZER_DIR.exists():
        tok_dest = profile_dir / "tokenizer"
        print(f"  Saving tokenizer -> {tok_dest}")
        shutil.copytree(TOKENIZER_DIR, tok_dest)

    fingerprint = _fingerprint_data_dir(profile_dir / "data")

    meta = {
        "dataset": name,
        "created": datetime.now().isoformat(),
        "shards": len(list((profile_dir / "data").glob("*.parquet"))) if (profile_dir / "data").exists() else 0,
        "fingerprint": fingerprint,
    }
    with open(profile_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if fingerprint:
        print(f"  Profile '{name}' saved (fingerprint: {fingerprint['hash']})")
    else:
        print(f"  Profile '{name}' saved (no fingerprint -- verify manually)")


def load_profile(name):
    """Restore a named profile as the active data + tokenizer."""
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        return False

    if not _validate_fingerprint(profile_dir, name):
        print(f"  Profile '{name}' failed validation -- will not load")
        print(f"  Run with --rebuild-profiles to fix")
        return False

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)

    data_src = profile_dir / "data"
    if data_src.exists():
        shutil.copytree(data_src, DATA_DIR)

    tok_src = profile_dir / "tokenizer"
    if tok_src.exists():
        shutil.copytree(tok_src, TOKENIZER_DIR)

    loaded_fp = _fingerprint_data_dir(DATA_DIR)
    with open(profile_dir / "meta.json") as f:
        meta = json.load(f)
    stored_fp = meta.get("fingerprint", {})

    if loaded_fp and stored_fp and loaded_fp["hash"] == stored_fp["hash"]:
        print(f"  Loaded profile '{name}' (verified: {loaded_fp['hash']})")
    else:
        print(f"  Loaded profile '{name}' (fingerprint verification skipped)")

    return True


def profile_exists(name):
    """Check if a valid profile exists."""
    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        return False
    if not (profile_dir / "data").exists():
        return False
    meta_path = profile_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("fingerprint"):
            return False
    return True


def delete_profile(name):
    """Delete a profile."""
    profile_dir = PROFILES_DIR / name
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        print(f"  Deleted profile '{name}'")


def list_profiles():
    """List all saved profiles with metadata and validation status."""
    if not PROFILES_DIR.exists():
        return []

    profiles = []
    for d in sorted(PROFILES_DIR.iterdir()):
        if d.is_dir():
            meta_path = d / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            fp = meta.get("fingerprint", {})
            valid = _validate_fingerprint(d, d.name) if fp else False
            profiles.append({
                "name": d.name,
                "shards": meta.get("shards", "?"),
                "created": meta.get("created", "unknown"),
                "fingerprint": fp.get("hash", "none"),
                "sample": fp.get("sample", "")[:60],
                "valid": valid,
            })
    return profiles


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_climbmix(num_shards=10):
    """Prepare the default climbmix dataset."""
    if profile_exists("climbmix"):
        print("  climbmix profile exists, loading...")
        if load_profile("climbmix"):
            return True
        else:
            print("  climbmix profile failed validation, rebuilding...")
            delete_profile("climbmix")

    print("  Downloading climbmix shards (fresh)...")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)

    result = subprocess.run(
        ["python", "-u", "prepare.py", f"--num-shards={num_shards}"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode == 0:
        fp = _fingerprint_data_dir(DATA_DIR)
        if fp:
            print(f"  Downloaded data sample: {fp['sample'][:80]}...")
        save_profile("climbmix", force=True)
        return True
    else:
        print(f"  Download failed")
        return False


def prepare_alternative(dataset_name, num_shards=10, num_source=3):
    """Prepare an alternative dataset via convert_dataset.py."""
    if profile_exists(dataset_name):
        print(f"  {dataset_name} profile exists, loading...")
        if load_profile(dataset_name):
            return True
        else:
            print(f"  {dataset_name} profile failed validation, rebuilding...")
            delete_profile(dataset_name)

    print(f"  Converting {dataset_name}...")
    result = subprocess.run(
        [
            "python", "-u", "convert_dataset.py", dataset_name,
            f"--num-shards={num_shards}",
            f"--num-source={num_source}",
            "--skip-backup",
        ],
        cwd=PROJECT_ROOT,
        timeout=3600,
    )
    if result.returncode != 0:
        print(f"  Conversion failed for {dataset_name}")
        return False

    print(f"  Training tokenizer for {dataset_name}...")
    result = subprocess.run(
        ["python", "-u", "prepare.py", "--num-shards=0"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"  Tokenizer training failed for {dataset_name}")
        return False

    fp = _fingerprint_data_dir(DATA_DIR)
    if fp:
        print(f"  Converted data sample: {fp['sample'][:80]}...")
    save_profile(dataset_name, force=True)
    return True


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def _model_slug(model: str | None) -> str | None:
    """Convert a model ID to a short directory-safe slug, or None for default."""
    from tui.llm_backend import ClaudeBackend
    if model is None or model == ClaudeBackend.DEFAULT_MODEL:
        return None
    slug = model.replace("claude-", "")
    return slug


def _write_deployment_manifest(results_dir, tag: str):
    """Write a manifest.json with hardware provenance for this deployment.

    Creates a tamper-evident record of which accelerator produced the results
    in this directory. Used by rsync scripts to validate data integrity.
    """
    manifest_path = results_dir / "manifest.json"

    try:
        from tui.hardware import get_hardware_summary
        hw = get_hardware_summary()
        gpu_name = hw.get("chip_name", "unknown")
        hbm_gb = round(hw.get("total_memory_gb", 0), 1)
        tpcs = hw.get("gpu_cores", 0)
        device_count = hw.get("device_count", 0)
        peak_tflops = hw.get("peak_tflops", 0)
    except Exception:
        gpu_name = "unknown"
        hbm_gb = 0
        tpcs = 0
        device_count = 0
        peak_tflops = 0

    # Try torch.hpu for runtime device count if hardware detection missed it
    if device_count == 0:
        try:
            import torch
            if hasattr(torch, "hpu") and torch.hpu.is_available():
                device_count = torch.hpu.device_count()
        except Exception:
            pass

    manifest = {
        "gpu_name": gpu_name,
        "hbm_gb": hbm_gb,
        "tpcs_per_device": tpcs,
        "device_count": device_count,
        "peak_tflops_bf16": peak_tflops,
        "tier": "gaudi3",
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(results_dir),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path} ({gpu_name}, {hbm_gb} GB HBM2e)")


def get_results_dir(dataset_name, model: str | None = None):
    """Get the results directory for a dataset, isolated by model."""
    slug = _model_slug(model)
    if slug:
        d = RESULTS_DIR / slug / dataset_name
    else:
        d = RESULTS_DIR / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_results(dataset_name, model: str | None = None):
    """Check if a dataset already has experiment results."""
    results_dir = get_results_dir(dataset_name, model)
    tsv = results_dir / "results.tsv"
    if not tsv.exists():
        return False
    with open(tsv) as f:
        lines = [l for l in f if l.strip() and not l.startswith("exp\t")]
    return len(lines) > 0


def count_experiments(dataset_name, model: str | None = None):
    """Count completed experiments for a dataset."""
    results_dir = get_results_dir(dataset_name, model)
    tsv = results_dir / "results.tsv"
    if not tsv.exists():
        return 0
    with open(tsv) as f:
        return sum(1 for l in f if l.strip() and not l.startswith("exp\t"))


def run_agent(dataset_name, tag, max_experiments=80, model=None):
    """Run the autonomous agent for a dataset (headless, no TUI)."""
    from tui.headless import run_headless

    results_dir = get_results_dir(dataset_name, model)
    results_tsv = str(results_dir / "results.tsv")
    run_tag = f"{tag}-{dataset_name}"

    # Write deployment manifest -- hardware provenance for this run
    _write_deployment_manifest(results_dir, tag)

    model_display = model or os.environ.get("CLAUDE_MODEL") or "default"
    slug = _model_slug(model)
    print(f"\n{'='*60}")
    print(f"  Running agent (headless): {dataset_name}")
    print(f"  Tag: {run_tag}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Model: {model_display}")
    print(f"  Results: {results_tsv}")
    if slug:
        print(f"  Isolation: results/{slug}/{dataset_name}/")
    print(f"{'='*60}\n")

    try:
        return run_headless(
            training_script="train_gaudi.py",
            results_path=results_tsv,
            tag=run_tag,
            max_experiments=max_experiments,
            model=model,
        )
    except KeyboardInterrupt:
        print(f"\n  Agent interrupted by user")
        return False


# ---------------------------------------------------------------------------
# Status and reporting
# ---------------------------------------------------------------------------

def _best_val_bpb_from_tsv(tsv_path):
    """Extract best val_bpb from a results.tsv file."""
    if not tsv_path.exists():
        return None
    with open(tsv_path) as f:
        vals = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8 and parts[7] in ("keep", "baseline"):
                try:
                    vals.append(float(parts[2]))
                except ValueError:
                    pass
    return min(vals) if vals else None


def print_status():
    """Print status of all datasets, including model-specific results."""
    print("\n  Multi-Dataset Experiment Status (Gaudi 3)")
    print("  " + "=" * 70)
    print(f"  {'Dataset':<20} {'Model':<16} {'Profile':<10} {'Experiments':<12} {'Best val_bpb':<14}")
    print("  " + "-" * 70)

    for name in DATASET_ORDER:
        has_profile = "yes" if profile_exists(name) else "no"

        n_exp = count_experiments(name)
        best = _best_val_bpb_from_tsv(RESULTS_DIR / name / "results.tsv")
        best_str = f"{best:.6f}" if best else "---"
        print(f"  {name:<20} {'default':<16} {has_profile:<10} {n_exp:<12} {best_str:<14}")

        if RESULTS_DIR.exists():
            for model_dir in sorted(RESULTS_DIR.iterdir()):
                if model_dir.is_dir() and (model_dir / name).is_dir():
                    slug = model_dir.name
                    if slug in [d for d in DATASET_ORDER]:
                        continue
                    tsv = model_dir / name / "results.tsv"
                    n = count_experiments(name, f"claude-{slug}")
                    if n > 0:
                        best_m = _best_val_bpb_from_tsv(tsv)
                        best_m_str = f"{best_m:.6f}" if best_m else "---"
                        print(f"  {'':<20} {slug:<16} {'':<10} {n:<12} {best_m_str:<14}")

    print("  " + "=" * 70)

    profiles = list_profiles()
    if profiles:
        print(f"\n  Saved profiles ({PROFILES_DIR}):")
        for p in profiles:
            status = "OK" if p["valid"] else "INVALID"
            fp = p.get("fingerprint", "none")
            print(f"    [{status}] {p['name']}: {p['shards']} shards, fp={fp}, created {p['created'][:10]}")
            if p.get("sample"):
                print(f"      Sample: {p['sample']}...")
    print()


# ---------------------------------------------------------------------------
# PID file locking — prevent duplicate suite runs
# ---------------------------------------------------------------------------

PIDFILE = PROJECT_ROOT / ".suite.pid"


def _acquire_pidlock() -> bool:
    """Write our PID to the lock file. Returns False if another suite is running."""
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
            # Check if the old process is still alive
            try:
                os.kill(old_pid, 0)  # signal 0 = existence check
                # Process exists — is it actually a run_suite.py?
                import platform
                if platform.system() == "Linux":
                    cmdline_path = f"/proc/{old_pid}/cmdline"
                    if os.path.exists(cmdline_path):
                        with open(cmdline_path) as f:
                            cmdline = f.read()
                        if "run_suite" in cmdline:
                            print(f"\n  ❌ ERROR: Another run_suite.py is already running (PID {old_pid})")
                            print(f"     Kill it first:  kill {old_pid}")
                            print(f"     Or force:       rm {PIDFILE} && re-run\n")
                            return False
                        # PID exists but isn't run_suite — stale lock
                    else:
                        # No /proc entry — stale lock
                        pass
                else:
                    # macOS / other: PID is alive, assume it's a suite
                    print(f"\n  ❌ ERROR: Another run_suite.py may be running (PID {old_pid})")
                    print(f"     Kill it first:  kill {old_pid}")
                    print(f"     Or force:       rm {PIDFILE} && re-run\n")
                    return False
            except ProcessLookupError:
                # PID doesn't exist — stale lock file
                pass
            except PermissionError:
                # Process exists but we can't signal it — assume alive
                print(f"\n  ❌ ERROR: Another run_suite.py may be running (PID {old_pid})")
                print(f"     Kill it first:  kill {old_pid}")
                print(f"     Or force:       rm {PIDFILE} && re-run\n")
                return False
        except (ValueError, OSError):
            # Corrupt PID file — remove it
            pass

    PIDFILE.write_text(str(os.getpid()))
    return True


def _release_pidlock():
    """Remove the PID lock file."""
    try:
        if PIDFILE.exists():
            # Only remove if it's still our PID
            stored = int(PIDFILE.read_text().strip())
            if stored == os.getpid():
                PIDFILE.unlink()
    except (ValueError, OSError):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset experiment suite runner (Gaudi 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=DATASET_ORDER,
                        help="Run a single dataset (default: run all in order)")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all datasets")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip datasets that already have results")
    parser.add_argument("--max-experiments", type=int, default=80,
                        help="Max experiments per dataset (default: 80)")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Training shards per dataset (default: 10)")
    parser.add_argument("--num-source", type=int, default=3,
                        help="Source files to download per dataset (default: 3)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag (default: today's date)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare datasets, don't run experiments")
    parser.add_argument("--save-profile", type=str, metavar="NAME",
                        help="Save current data+tokenizer as a named profile")
    parser.add_argument("--load-profile", type=str, metavar="NAME",
                        help="Load a named profile as active data+tokenizer")
    parser.add_argument("--rebuild-profiles", action="store_true",
                        help="Delete and rebuild all profiles from scratch")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all profiles and report any mismatches")
    parser.add_argument("--model", type=str, default=None,
                        help="Claude model override (e.g. claude-sonnet-4-6)")

    args = parser.parse_args()

    if args.save_profile:
        print(f"Saving profile '{args.save_profile}'...")
        save_profile(args.save_profile)
        return

    if args.load_profile:
        print(f"Loading profile '{args.load_profile}'...")
        if load_profile(args.load_profile):
            print("Done! Ready to train.")
        else:
            print(f"ERROR: Profile '{args.load_profile}' not found.")
            sys.exit(1)
        return

    if args.validate:
        print("\n  Profile Validation")
        print("  " + "=" * 70)
        profiles = list_profiles()
        if not profiles:
            print("  No profiles found.")
        else:
            for p in profiles:
                status = "VALID" if p["valid"] else "INVALID"
                fp = p["fingerprint"]
                print(f"  {p['name']:<20} {status:<12} fp={fp:<18} {p['sample']}")
        print("  " + "=" * 70)
        return

    if args.rebuild_profiles:
        print("\n  Rebuilding all profiles...")
        if PROFILES_DIR.exists():
            for d in PROFILES_DIR.iterdir():
                if d.is_dir():
                    print(f"  Deleting profile '{d.name}'...")
                    shutil.rmtree(d)
        print("  All profiles deleted. They will be rebuilt on next suite run.")
        print("  Run: python run_suite.py --prepare-only")
        return

    if args.status:
        print_status()
        return

    # --- PID lock (prevent duplicate suite runs) ---
    if not _acquire_pidlock():
        sys.exit(1)
    import atexit
    atexit.register(_release_pidlock)

    tag = args.tag or datetime.now().strftime("%b%d").lower()
    datasets = [args.dataset] if args.dataset else DATASET_ORDER

    model_display = args.model or os.environ.get("CLAUDE_MODEL") or "default"
    print(f"\nMulti-Dataset Experiment Suite (Gaudi 3)")
    print(f"  Tag: {tag}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max experiments per dataset: {args.max_experiments}")
    print(f"  Shards per dataset: {args.num_shards}")
    print(f"  Model: {model_display}")
    print()

    for i, dataset_name in enumerate(datasets):
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(datasets)}] Dataset: {dataset_name}")
        print(f"{'#'*60}")

        if args.skip_completed and has_results(dataset_name, args.model):
            n = count_experiments(dataset_name, args.model)
            slug = _model_slug(args.model)
            loc = f" (model: {slug})" if slug else ""
            print(f"  Skipping -- already has {n} experiments{loc}")
            continue

        print(f"\n  Preparing {dataset_name}...")
        if dataset_name == "climbmix":
            success = prepare_climbmix(args.num_shards)
        else:
            num_source = args.num_source
            if dataset_name == "slimpajama":
                num_source = max(6, args.num_source)
            elif dataset_name == "github-code-python":
                num_source = max(5, args.num_source)
            success = prepare_alternative(dataset_name, args.num_shards, num_source)

        if not success:
            print(f"  FAILED to prepare {dataset_name}, skipping")
            continue

        if args.prepare_only:
            print(f"  Prepared {dataset_name} (--prepare-only, skipping agent run)")
            continue

        run_agent(dataset_name, tag, args.max_experiments, model=args.model)

    print_status()


if __name__ == "__main__":
    main()
