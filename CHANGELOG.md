# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [3.0.1] - 2026-10-02

### Fixed
- Frames are read until the video ends instead of up to the container's frame count, and the GPU path processes its last partial batch. Before, a wrong count could drop up to `batch_size - 1` frames on the GPU, or every frame past the count on both paths ([#16](https://github.com/joeljose/Visual-Mic/issues/16), [#29](https://github.com/joeljose/Visual-Mic/pull/29)).
- Filter settings are checked before processing: a cutoff of zero or less, or `-fl` at or above the Nyquist frequency, is an error instead of a crash or a silent skip at the end. Clips shorter than 28 frames are filtered with a shorter edge pad instead of crashing. A frame rate below 500 fps prints a warning that `--fps` is probably needed. The WAV sample rate is rounded, not truncated ([#15](https://github.com/joeljose/Visual-Mic/issues/15), [#29](https://github.com/joeljose/Visual-Mic/pull/29)).

## [3.0.0] - 2026-10-02

### Changed
- The output differs from 2.0.0 ([#27](https://github.com/joeljose/Visual-Mic/pull/27)). Sub-bands are now combined into horizontal and vertical motion and projected onto the main vibration direction, instead of being aligned by cross-correlation and summed. On the MIT Chips2 video the old alignment picked lags as long as the whole clip. SNR against the played sound goes from -11.1 dB to -3.8 dB. Rigid vertical motion, which half the orientations see upside down, now comes through (synthetic test: -19.5 dB with plain summing, 16 dB now).
- A high-pass filter is on by default, at fps/40 kept within 20 to 100 Hz, as in Davis et al. `--no-filter` turns it off ([#14](https://github.com/joeljose/Visual-Mic/issues/14)).
- The CPU image uses `opencv-python-headless` and no apt packages (1.09 GB to 757 MB).
- The GPU image is built on `python:3.11-slim` with the torch 2.1.2 CUDA wheels instead of `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` (about 2 GB smaller unpacked).
- `pytorch_wavelets` is pinned to a commit.
- Images use a fixed non-root user (uid 1000) instead of `UID`/`GID`/`UNAME` build args. Run with `--user "$(id -u):$(id -g)"` for bind mounts.
- CI runs pytest. The GPU image is built only when its inputs change.

### Added
- `--denoise`: spectral subtraction with a median noise estimate. It removes steady hum and light flicker (Chips2: SNR -2.9 dB, segSNR -1.3 dB; MIT's published result: -4.0 dB, -1.1 dB).
- `scripts/eval_audio.py` scores recovered audio against the played sound (SNR, segmental SNR, coherence).
- Synthetic recovery tests with known sub-pixel motion (`tests/test_recovery.py`, [#26](https://github.com/joeljose/Visual-Mic/pull/26)).

### Removed
- `find_best_shift` and the `ref_level`/`ref_orient` parameters.

### Fixed
- The CPU install pulled NumPy 2, which breaks `dtcwt` (`np.asfarray` was removed). NumPy is now `<2`.
- CI broke on ruff 0.16. Dev tools are now pinned (`ruff==0.15.7`, `pytest==8.4.2`) ([#25](https://github.com/joeljose/Visual-Mic/pull/25)).
- The `--version` test reads `VERSION`, so `VERSION` and `__version__` can't drift apart.

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
