# Design Doc: Visual-Mic Project Hardening

**Status: APPROVED**

## Context

The Visual-Mic project is a Python implementation of the Visual Microphone algorithm (Davis et al., SIGGRAPH 2014) that recovers sound from high-speed video using 2D DTCWT phase extraction. It's a single-file CLI tool (`visualmic.py`) with CPU and GPU paths, shipped via Docker.

A code review identified gaps vs the sibling repos (EVM, DTCWT Motion Mag): no unit tests, no CI/CD, no versioning, no design docs, duplicated denoising code, outdated GPU Dockerfile, and hard-coded parameters. This design doc covers the hardening plan.

**PRD**: https://github.com/joeljose/Visual-Mic/issues/2

## Goals and Non-Goals

**Goals:**
- Remove denoising code (duplicates `joeljose/audio_denoising` repo)
- Add unit tests for CPU and GPU codepaths
- Establish versioning infrastructure (VERSION file, `__version__`, `--version`, Docker labels)
- Add `--nlevels`, `--biort`, `--qshift` CLI flags
- Add pre-flight memory/VRAM estimation
- Add CI/CD pipeline (lint + import + help + version)
- Add changelog, contributing guide, design doc, dev dependencies
- Upgrade GPU Dockerfile to PyTorch 2.1.2 / CUDA 12.1
- Clean up README (remove denoising, add Development section, restructure Setup)

**Non-Goals:**
- Refactoring GPU/CPU into shared code paths (defer — `postprocess_phase_signals` already shared)
- Multiprocessing across frames (defer)
- Refactoring into a pip-installable package
- GPU testing in CI (no hosted GPU runners)
- Integration tests with audio output comparison

## Proposed Design

### A. Remove Denoising Code

Delete from `visualmic.py`:
- `denoise_spectral()` (lines 37–57) and `denoise_morphological()` (lines 60–86)
- `--denoise` and `--denoise-input` CLI arguments
- Standalone denoise mode block (lines 321–340)
- Post-pipeline denoise application (lines 448–454)
- `from scipy.io.wavfile import read as read_wav` (only needed for `--denoise-input`)
- `from scipy import ndimage` (only needed for morphological denoising)
- Update error message at line 344 to remove `--denoise-input` reference

The `audio_denoising` repo at `github.com/joeljose/audio_denoising` provides this functionality separately.

### B. CLI Flag Additions

**`--nlevels` (default: 3):**
Currently hard-coded at line 419. Make it a CLI argument. Validation: must be >= 1. Update `min_dim = 2 ** nlevels` (line 420) to use the argument.

**`--biort` (default: `near_sym_b`) and `--qshift` (default: `qshift_b`):**
Matching DTCWT Motion Mag v2.0.0 defaults. Applied to both CPU and GPU paths:
- CPU: `transform.forward(gray, nlevels=nlevels, biort=biort, qshift=qshift)` — note: `dtcwt.Transform2d()` accepts biort/qshift in constructor
- GPU: `DTCWTForward(J=nlevels, biort=biort, qshift=qshift)` — already parameterized in the code (line 202), just needs to read from args instead of hard-coding

Available options (same as DTCWT Motion Mag):
- biort: `antonini`, `legall`, `near_sym_a`, `near_sym_b`
- qshift: `qshift_06`, `qshift_a`, `qshift_b`, `qshift_c`, `qshift_d`

**`--version`:**
Standard argparse version action reading from `__version__`.

### C. Pre-Flight Memory Estimation

**CPU path:**
Visual-Mic is memory-efficient — it streams frames and only stores the phase signal array `(num_frames, nlevels, n_orient)` which is small (e.g., 301 × 3 × 6 × 8 bytes = 43 KB). The main memory consumer is the reference conjugate coefficients (one frame's worth of DTCWT coefficients). No pre-flight check needed for CPU — it can handle arbitrarily long videos.

**GPU path:**
The GPU path batches frames, so peak VRAM depends on batch size:
```
vram_per_frame = H × W × 4 bytes (float32 input)
dtcwt_overhead = ~15× per frame (forward transform intermediates)
vram_estimate = batch_size × H × W × 4 × 15 + 300 MB (PyTorch overhead)
```

Add `estimate_vram()` function. Before processing, query `torch.cuda.mem_get_info()`, compare against estimate, warn if > 70% of available VRAM. Suggest reducing `--batch-size` if insufficient.

Also add OOM error handling in the GPU batch loop (already exists at line 236, just needs consistent messaging matching the other repos):
```
Error: GPU out of memory with batch_size={batch_size}.
  Suggestions:
  - Reduce --batch-size (current: {batch_size})
  - Use --roi to crop to a smaller region
  - Remove --gpu to use CPU mode
```

### D. Versioning

Same pattern as EVM and DTCWT Motion Mag:
- `VERSION` file at repo root containing `2.0.0`
- `__version__ = "2.0.0"` baked into `visualmic.py`
- `--version` CLI flag via argparse
- Docker build scripts read `VERSION`, tag images as `visual-mic:{version}` + `visual-mic:latest`
- Dockerfiles receive `--build-arg VERSION` and apply `LABEL version=${VERSION}`

### E. Testing Infrastructure

**`requirements-dev.txt`:**
```
pytest>=7.0,<9
ruff>=0.4.0,<1
```

**`tests/test_visualmic.py`** — CPU tests:

Tier 1 (strict, exact equality):
- `TestFormatDuration`: 0s, 59s, 60s, 3661s
- `TestFindBestShift`: known shift on synthetic signals (sine wave shifted by N samples)
- `TestSaveWav`: output file exists, correct sample rate, int16 range

Tier 2 (moderate tolerance):
- `TestPostprocessPhaseSignals`: constant input → silent output, normalization to [-1, 1], cross-correlation alignment on known-shifted synthetic signals
- `TestButterworthFilter`: in-band signal preserved, out-of-band signal attenuated, freq_low >= Nyquist warning
- `TestEstimateVram`: arithmetic correctness, batch size scaling

Tier 3 (smoke):
- `TestExtractAudio`: synthetic 256×256 video (random noise, 32 frames), verify output shape = (32,), all values finite, values in [-1, 1]
- `TestInputValidation`: missing file, invalid frequencies, bad ROI format, ROI out of bounds, invalid nlevels

**`tests/test_visualmic_gpu.py`** — GPU tests:

All wrapped in `@pytest.mark.skipif(not HAS_CUDA)`:
- `TestGpuForwardPass`: DTCWTForward on 256×256 batch, verify Yh shapes and finite values
- `TestExtractAudioGpu`: smoke test on synthetic 256×256 video, verify output shape and finiteness
- `TestEstimateVram`: VRAM arithmetic

**`test.sh`:**
Matches DTCWT Motion Mag pattern — Docker-based, supports `cpu`/`gpu` modes:
```bash
./test.sh          # CPU lint + tests
./test.sh gpu      # GPU lint + tests
./test.sh --build  # Force rebuild
```

### F. CI/CD

**`.github/workflows/ci.yml`:**
Two jobs matching EVM/DTCWT pattern:

CPU job:
1. Build Docker image (CPU Dockerfile)
2. Lint (`ruff check .`)
3. Verify import (`python -c "import visualmic"`)
4. Verify `--help`
5. Verify `--version`

GPU job:
1. Build Docker image (GPU Dockerfile)
2. Lint (`ruff check .`)
3. Verify import
4. Verify `--help`
5. Verify `--version`

No pipeline smoke test in CI (no test video committed to repo — MIT CSAIL videos are 14 GB).

**Manual GPU verification (post-implementation):**
Download a MIT CSAIL test video (e.g., `Chips1-2200Hz-Mary_Had-input.avi` from http://data.csail.mit.edu/vidmag/VisualMic/Results/) and run the GPU pipeline end-to-end to verify audio output is recognizable. CPU verification with real videos is impractical — the MIT CSAIL videos are 704×704 × 22,859 frames, which would take too long or OOM on CPU.

### G. GPU Dockerfile Upgrade

**From:** `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`
**To:** `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`

**Verified compatible:** Live-tested `pytorch_wavelets.DTCWTForward` with same API Visual-Mic uses (J=3, biort='near_sym_b', qshift='qshift_b', Yh[level][..., 0/1] indexing) on PyTorch 2.1.2 / CUDA 12.1. All outputs finite and correct shapes.

**`requirements-gpu.txt` changes:**
```
scipy>=1.7.1,<2
numpy>=1.20.3,<2    # <2 required: pytorch_wavelets uses removed NumPy 2.0 APIs
opencv-python-headless>=4.7.0,<5
PyWavelets>=1.1.0
```

Both Dockerfiles updated to:
- Copy `tests/` and `requirements-dev.txt`
- Accept `--build-arg VERSION` and apply `LABEL version=${VERSION}`

### H. Documentation

**`CHANGELOG.md`:**
Keep a Changelog format, starting at v2.0.0 (no backfill):
```markdown
# Changelog

## [2.0.0] - YYYY-MM-DD
### Removed
- Denoising (`--denoise`, `--denoise-input`). Use `audio_denoising` repo instead.

### Added
- `--nlevels`, `--biort`, `--qshift` CLI flags
- `--version` flag
- Pre-flight GPU memory estimation
- Unit tests (CPU + GPU)
- CI/CD pipeline
- CHANGELOG.md, CONTRIBUTING.md, design docs

### Changed
- Default wavelet filters: `near_sym_b`/`qshift_b` (both CPU and GPU paths)
- GPU Dockerfile upgraded to PyTorch 2.1.2 / CUDA 12.1
- Docker images now tagged with version numbers
```

**`CONTRIBUTING.md`:**
Same structure as EVM/DTCWT repos:
- Open issue first
- Fork and branch from main
- Small PRs, one logical change
- PEP 8 + ruff
- Run `./test.sh` before opening PR

**README changes:**
- Remove Part 4 (Denoising) entirely
- Restructure Setup section A (remove YouTube link, match terse style of other repos)
- Add CI badge at top
- Add Development section (Running Tests, Versioning, Project Structure)
- Update CLI flags table with `--nlevels`, `--biort`, `--qshift`, `--version`
- Update Future Work: remove "GPU-accelerated DTCWT" (done), remove denoising reference
- Add project structure tree

## Alternatives Considered

### Remove denoising vs keep as optional

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Remove entirely | Single responsibility, less code to maintain, separate repo exists | Users lose integrated denoising | **Chosen** — `audio_denoising` repo is the right place for this |
| Keep as optional | Convenience for users | Duplicated code, scope creep, adds scipy.ndimage dependency | Rejected |

### Default wavelet filters: near_sym_a vs near_sym_b

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Keep `near_sym_a` (dtcwt default) | Backward compatible | Inconsistent with GPU path (already uses `near_sym_b`), worse quality at high magnification | Rejected |
| Switch to `near_sym_b` | Matches DTCWT Motion Mag, matches existing GPU path, fewer artifacts | Breaking change for CPU users | **Chosen** — v2.0.0 justifies the break, `--biort`/`--qshift` give user control |

### Memory estimation: CPU + GPU vs GPU only

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Both CPU and GPU | Consistent with DTCWT Motion Mag | CPU path is streaming (tiny memory footprint), check is pointless | Rejected — would always pass |
| GPU only | Targets actual OOM risk | No CPU protection | **Chosen** — CPU path streams frames, only stores phase_signals array (~KB), OOM is not a realistic risk |

### Version: 1.1.0 vs 2.0.0

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| 1.1.0 | Lower number | Violates semver — removing CLI flags is breaking | Rejected |
| 2.0.0 | Semver correct, matches DTCWT precedent | Higher number | **Chosen** — breaking changes require major bump |

## Tradeoffs and Risks

- **`pytorch_wavelets` is unmaintained** (last commit 2023). Uses stable PyTorch APIs (`F.conv2d`, `autograd.Function`), but `pkg_resources` will break on Python 3.14+. Mitigated: pin PyTorch 2.1.2 and numpy<2 in Docker. Can fork if needed.

- **CPU/GPU outputs differ.** Different DTCWT implementations (`dtcwt` vs `pytorch_wavelets`), different precision (float64 vs float32). Both produce valid audio; they are not cross-comparable. Documented, accepted.

- **No pipeline smoke test in CI.** Unlike EVM/DTCWT repos (which have `face.mp4` committed), Visual-Mic has no small test video in the repo. Mitigated: synthetic video tests run locally via `./test.sh`, and manual GPU verification against MIT CSAIL videos post-implementation.

- **CPU path untested with real videos.** MIT CSAIL videos (704×704, 22K+ frames) are too large for CPU processing in reasonable time or memory. GPU path is the only practical way to verify against real data.

- **GPU memory estimation is approximate.** PyTorch's actual VRAM usage depends on allocator behavior, fragmentation, and cuDNN workspace sizes. The 70% threshold provides safety margin.

- **Default filter change is breaking for CPU path.** CPU path previously used `dtcwt` defaults (`near_sym_a`/`qshift_a`), now explicitly uses `near_sym_b`/`qshift_b`. Users can restore old behavior with `--biort near_sym_a --qshift qshift_a`.

## Open Questions

None — all decisions resolved during PRD and grill phases.
