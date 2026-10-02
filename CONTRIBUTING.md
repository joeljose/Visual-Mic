# Contributing

Thanks for your interest in contributing!

## How to contribute

1. **Open an issue first.** Describe the bug or feature before writing code. No silent PRs.
2. **Fork and branch.** Create a feature branch from `main`.
3. **Keep PRs small.** One logical change per PR.
4. **Follow PEP 8.** CI checks it with `ruff`.
5. **Test before opening a PR.** Run `./test.sh`, and `./test.sh gpu` if you touch GPU code.

## Running tests

All tests run inside Docker, so you need no local Python dependencies:

```bash
./test.sh          # CPU: lint + unit tests
./test.sh gpu      # GPU: lint + unit tests (requires nvidia-container-toolkit)
./test.sh --build  # Force rebuild image before testing
```

## Dependencies

`requirements*.txt` hold the allowed version ranges. The Docker images install from the hash-locked `requirements.lock` (CPU) and `requirements-gpu.lock` (GPU), so a build today and a build next year get the same packages. After changing a `requirements*.txt` file, or to pick up new releases on purpose, regenerate both locks with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile requirements.txt requirements-dev.txt --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements.lock
uv pip compile requirements-gpu.txt requirements-dev.txt --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements-gpu.lock
```

The base image is pinned by digest, GitHub Actions by commit, and `pytorch_wavelets` by commit in `Dockerfile.gpu`. Dependabot proposes updates for the actions and the base image once a month.

