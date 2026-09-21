

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil


EXPECTED_PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIRNAME = "results"
HF_CACHE_DIRNAME = "hf_cache"


def safe_import_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def bytes_to_gb(value: int | float) -> float:
    return round(value / (1024 ** 3), 3)


def get_cpu_name() -> str:
    name = platform.processor().strip()

    if name:
        return name

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()

    except Exception:
        pass

    return "unknown"


def directory_size_bytes(path: Path) -> int:
    total = 0

    if not path.exists():
        return 0

    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue

    return total


def main() -> int:
    project_dir = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else EXPECTED_PROJECT
    )

    output_dir = project_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    disk = shutil.disk_usage(
        project_dir if project_dir.exists() else Path.cwd()
    )

    cpu_freq = psutil.cpu_freq()

    profile = {
        "timestamp_local": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),

        "project_dir": str(project_dir),

        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },

        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },

        "cpu": {
            "name": get_cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_percent": psutil.cpu_percent(interval=1.0),
            "frequency_mhz_current": (
                round(cpu_freq.current, 1) if cpu_freq else None
            ),
            "frequency_mhz_min": (
                round(cpu_freq.min, 1) if cpu_freq else None
            ),
            "frequency_mhz_max": (
                round(cpu_freq.max, 1) if cpu_freq else None
            ),
        },

        "ram": {
            "total_gb": bytes_to_gb(vm.total),
            "available_gb": bytes_to_gb(vm.available),
            "used_gb": bytes_to_gb(vm.used),
            "percent_used": round(vm.percent, 2),
            "free_gb": bytes_to_gb(vm.free),
        },

        "swap": {
            "total_gb": bytes_to_gb(swap.total),
            "used_gb": bytes_to_gb(swap.used),
            "free_gb": bytes_to_gb(swap.free),
            "percent_used": round(swap.percent, 2),
        },

        "disk": {
            "path": str(
                project_dir if project_dir.exists() else Path.cwd()
            ),
            "total_gb": bytes_to_gb(disk.total),
            "used_gb": bytes_to_gb(disk.used),
            "free_gb": bytes_to_gb(disk.free),
        },

        "packages": {
            "torch": safe_import_version("torch"),
            "transformers": safe_import_version("transformers"),
            "sentence_transformers": safe_import_version(
                "sentence_transformers"
            ),
            "numpy": safe_import_version("numpy"),
            "pandas": safe_import_version("pandas"),
            "scikit_learn": safe_import_version("sklearn"),
            "psutil": safe_import_version("psutil"),
            "tabulate": safe_import_version("tabulate"),
        },
    }

    # ---------------------------------------------------------
    # PyTorch runtime
    # ---------------------------------------------------------
    try:
        import torch

        profile["pytorch_runtime"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": (
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(
                torch.get_num_interop_threads()
            ),
        }

        if torch.cuda.is_available():
            profile["pytorch_runtime"]["gpu_name"] = (
                torch.cuda.get_device_name(0)
            )

            profile["pytorch_runtime"]["gpu_memory_gb"] = bytes_to_gb(
                torch.cuda.get_device_properties(0).total_memory
            )

    except Exception as exc:
        profile["pytorch_runtime"] = {
            "error": f"{type(exc).__name__}: {exc}"
        }

    # ---------------------------------------------------------
    # Current process memory
    # ---------------------------------------------------------
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    profile["current_process_memory"] = {
        "rss_gb": bytes_to_gb(memory_info.rss),
        "vms_gb": bytes_to_gb(memory_info.vms),
        "note": (
            "Current profile process only; not peak memory "
            "of the 500-question experiments."
        ),
    }

    # ---------------------------------------------------------
    # Hugging Face cache
    # ---------------------------------------------------------
    hf_cache = project_dir / HF_CACHE_DIRNAME

    profile["project_cache"] = {
        "hf_cache_exists": hf_cache.exists(),
        "hf_cache_size_gb": bytes_to_gb(
            directory_size_bytes(hf_cache)
        ),
        "hf_cache_path": str(hf_cache),
    }

    # ---------------------------------------------------------
    # Paper-relevant facts
    # ---------------------------------------------------------
    torch_info = profile.get("pytorch_runtime", {})

    profile["paper_facts"] = {
        "cpu_only": torch_info.get("device") == "cpu",
        "total_ram_gb": profile["ram"]["total_gb"],
        "physical_cores": profile["cpu"]["physical_cores"],
        "logical_cores": profile["cpu"]["logical_cores"],
        "pytorch_threads": torch_info.get("num_threads"),
        "pytorch_interop_threads": torch_info.get(
            "num_interop_threads"
        ),
        "cuda_available": torch_info.get("cuda_available"),
        "swap_used_gb_now": profile["swap"]["used_gb"],
        "disk_free_gb": profile["disk"]["free_gb"],
    }

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    json_path = output_dir / "system_profile.json"

    json_path.write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Flatten data for CSV
    # ---------------------------------------------------------
    flat_rows = []

    def flatten(data: dict, prefix: str = "") -> None:
        for key, value in data.items():
            full_key = (
                f"{prefix}.{key}" if prefix else key
            )

            if isinstance(value, dict):
                flatten(value, full_key)
            else:
                flat_rows.append((full_key, value))

    flatten(profile)

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------
    csv_path = output_dir / "system_profile.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "metric",
            "value",
        ])

        writer.writerows(flat_rows)

    # ---------------------------------------------------------
    # Print report
    # ---------------------------------------------------------
    print()
    print("=" * 72)
    print("TinyRAG SYSTEM / RUNTIME PROFILE")
    print("=" * 72)

    print(f"Project:          {project_dir}")
    print(f"OS:               {profile['os']['platform']}")
    print(f"CPU:              {profile['cpu']['name']}")

    print(
        f"CPU cores:        "
        f"{profile['cpu']['physical_cores']} physical / "
        f"{profile['cpu']['logical_cores']} logical"
    )

    print(
        f"RAM total:        "
        f"{profile['ram']['total_gb']} GB"
    )

    print(
        f"RAM available:    "
        f"{profile['ram']['available_gb']} GB"
    )

    print(
        f"RAM used:         "
        f"{profile['ram']['percent_used']}%"
    )

    print(
        f"Swap:             "
        f"{profile['swap']['total_gb']} GB total / "
        f"{profile['swap']['percent_used']}% used"
    )

    print(
        f"Disk free:        "
        f"{profile['disk']['free_gb']} GB"
    )

    print(
        f"Python:           "
        f"{profile['python']['version']}"
    )

    print(
        f"PyTorch:          "
        f"{profile['packages']['torch']}"
    )

    print(
        f"PyTorch device:   "
        f"{torch_info.get('device', 'unknown')}"
    )

    print(
        f"CUDA available:   "
        f"{torch_info.get('cuda_available', 'unknown')}"
    )

    print(
        f"Torch threads:    "
        f"{torch_info.get('num_threads', 'unknown')}"
    )

    print(
        f"Torch inter-op:   "
        f"{torch_info.get('num_interop_threads', 'unknown')}"
    )

    print(
        f"HF cache size:    "
        f"{profile['project_cache']['hf_cache_size_gb']} GB"
    )

    print(
        f"Current RSS:      "
        f"{profile['current_process_memory']['rss_gb']} GB"
    )

    print()
    print("PAPER-RELEVANT FACTS")
    print("-" * 72)

    for key, value in profile["paper_facts"].items():
        print(f"{key:24s}: {value}")

    print()
    print("PACKAGE VERSIONS")
    print("-" * 72)

    for key, value in profile["packages"].items():
        print(f"{key:24s}: {value}")

    print()
    print("OUTPUT FILES")
    print("-" * 72)

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")

    print()
    print(
        "NOTE: Current RAM is a machine snapshot. "
        "Keep the already-recorded experiment-level RAM "
        "values for the RAG/no-RAG results."
    )

    print("=" * 72)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())