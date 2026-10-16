#!/usr/bin/env python3
"""
Hardware/environment inspection script (Section 28).

Run this BEFORE choosing model_name/compute_type settings for a new
deployment target:

    python scripts/check_environment.py

It reports CPU, RAM, GPU/VRAM, CUDA availability and disk space, then
prints a suggested `.env` configuration for that hardware. It never
changes any files — it only prints suggestions for you to copy into `.env`.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def _try_import_torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:
        return None


def _cpu_info() -> dict:
    info = {"logical_cores": None, "model": None}
    try:
        info["logical_cores"] = len(__import__("os").sched_getaffinity(0))
    except AttributeError:
        import os

        info["logical_cores"] = os.cpu_count()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        info["model"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return info


def _ram_info() -> dict:
    try:
        with open("/proc/meminfo") as f:
            lines = {line.split(":")[0]: line.split(":")[1].strip() for line in f}
        total_kb = int(lines["MemTotal"].split()[0])
        return {"total_gb": round(total_kb / 1024 / 1024, 1)}
    except Exception:
        return {"total_gb": None}


def _disk_info(path: str = ".") -> dict:
    usage = shutil.disk_usage(path)
    return {"total_gb": round(usage.total / 1e9, 1), "free_gb": round(usage.free / 1e9, 1)}


def _gpu_info() -> dict:
    info = {"available": False, "name": None, "vram_gb": None, "cuda_version": None}
    torch = _try_import_torch()
    if torch is not None and torch.cuda.is_available():
        info["available"] = True
        info["name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
        return info
    # Fall back to nvidia-smi if torch isn't installed yet.
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                name, mem = out.stdout.strip().split(",")
                info["available"] = True
                info["name"] = name.strip()
                info["vram_gb"] = mem.strip()
        except Exception:
            pass
    return info


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def suggest_config(cpu: dict, ram: dict, gpu: dict, disk: dict) -> str:
    lines = []
    if gpu["available"]:
        lines.append("WHISPER_DEVICE=cuda")
        lines.append("WHISPER_COMPUTE_TYPE=float16")
        lines.append("DIARIZATION_DEVICE=cuda")
        lines.append("SUMMARIZER_DEVICE=cuda")
        vram = gpu.get("vram_gb")
        if isinstance(vram, (int, float)) and vram < 8:
            lines.append("WHISPER_MODEL_NAME=medium  # < 8GB VRAM: large-v3 may not fit comfortably")
        else:
            lines.append("WHISPER_MODEL_NAME=large-v3")
    else:
        lines.append("WHISPER_DEVICE=cpu")
        lines.append("WHISPER_COMPUTE_TYPE=int8")
        lines.append("DIARIZATION_DEVICE=cpu")
        lines.append("SUMMARIZER_DEVICE=cpu")
        ram_gb = ram.get("total_gb") or 0
        if ram_gb and ram_gb < 8:
            lines.append("WHISPER_MODEL_NAME=small  # <8GB RAM: large-v3 on CPU will be very slow/heavy")
            lines.append("SUMMARIZER_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct  # smaller model for low-RAM CPU boxes")
        elif ram_gb < 16:
            lines.append("WHISPER_MODEL_NAME=medium  # moderate RAM: large-v3 possible but slow on CPU")
        else:
            lines.append("WHISPER_MODEL_NAME=large-v3")
    return "\n".join(lines)


def main() -> int:
    cpu, ram, gpu, disk = _cpu_info(), _ram_info(), _gpu_info(), _disk_info()

    print("=== Environment report ===")
    print(f"Platform      : {platform.platform()}")
    print(f"CPU           : {cpu['model'] or 'unknown'} ({cpu['logical_cores']} logical cores)")
    print(f"RAM           : {ram['total_gb']} GB" if ram["total_gb"] else "RAM           : unknown")
    print(f"Disk (cwd)    : {disk['free_gb']} GB free / {disk['total_gb']} GB total")
    if gpu["available"]:
        print(f"GPU           : {gpu['name']} ({gpu['vram_gb']} GB VRAM, CUDA {gpu['cuda_version']})")
    else:
        print("GPU           : none detected (CPU-only mode)")
    print(f"ffmpeg        : {'found' if _ffmpeg_available() else 'NOT FOUND — install it before running the pipeline'}")

    print("\n=== Suggested .env overrides for this machine ===")
    print(suggest_config(cpu, ram, gpu, disk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
