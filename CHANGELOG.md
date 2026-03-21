# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
