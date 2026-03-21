# Contributing

Thanks for your interest in contributing!

## How to contribute

1. **Open an issue first** — describe the bug or feature before writing code. No silent PRs.
2. **Fork and branch** — create a feature branch from `main`.
3. **Keep PRs small** — one logical change per PR.
4. **Follow PEP 8** — enforced by `ruff` in CI.
5. **Test before opening a PR** — run `./test.sh` (and `./test.sh gpu` if touching GPU code).

## Running tests

All tests run inside Docker — no local Python dependencies needed:

```bash
./test.sh          # CPU: lint + unit tests
./test.sh gpu      # GPU: lint + unit tests (requires nvidia-container-toolkit)
./test.sh --build  # Force rebuild image before testing
```
