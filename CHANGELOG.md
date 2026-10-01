# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- CPU install pulled NumPy 2, which breaks `dtcwt` (`np.asfarray` removed); NumPy is now `<2`
- CI broke on ruff 0.16; dev tools are now pinned (`ruff==0.15.7`, `pytest==8.4.2`)

### Changed
- CPU image uses `opencv-python-headless` and no apt packages (1.09 GB to 757 MB)
- GPU image is built on `python:3.11-slim` with the torch 2.1.2 CUDA wheels instead of `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` (about 2 GB smaller unpacked)
- `pytorch_wavelets` is pinned to a commit
- Images use a fixed non-root user (uid 1000) instead of `UID`/`GID`/`UNAME` build args; run with `--user "$(id -u):$(id -g)"` for bind mounts
- CI runs pytest; the GPU image is built only when its inputs change

## [2.0.0] - 2026-03-21

### Removed
- Denoising (`--denoise`, `--denoise-input`). Use the standalone [`audio_denoising`](https://github.com/joeljose/audio_denoising) tool instead.

### Added
- `--nlevels` CLI flag for configurable DTCWT decomposition levels (default: 3)
- `--biort` and `--qshift` CLI flags for wavelet filter selection (default: `near_sym_b`/`qshift_b`)
- `--version` flag
- Pre-flight GPU VRAM estimation and warning
- Unit tests for CPU and GPU paths (`tests/test_visualmic.py`, `tests/test_visualmic_gpu.py`)
- Docker-based test runner (`test.sh`) supporting cpu/gpu modes
- CI/CD pipeline (`.github/workflows/ci.yml`)
- `CHANGELOG.md`, `CONTRIBUTING.md`, design docs
- `VERSION` file as single source of truth for versioning
- Docker images tagged with version numbers

### Changed
- Default wavelet filters changed to `near_sym_b`/`qshift_b` for both CPU and GPU paths (matching DTCWT Motion Mag v2.0.0). Restore old behavior with `--biort near_sym_a --qshift qshift_a`.
- GPU Dockerfile upgraded from PyTorch 1.12.1/CUDA 11.3 to PyTorch 2.1.2/CUDA 12.1
- `-i`/`--input` is now always required (previously optional when using `--denoise-input`)
- Pinned upper bounds on all dependencies
