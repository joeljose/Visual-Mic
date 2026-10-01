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
