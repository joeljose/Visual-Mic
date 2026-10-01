#!/bin/bash
set -e

MODE="${1:-cpu}"
BUILD_FLAG="${2:-}"

if [[ "$MODE" == "gpu" ]]; then
    IMAGE="visual-mic-gpu-dev"
    DOCKERFILE="-f Dockerfile.gpu"
    RUN_FLAGS="--gpus device=0"
elif [[ "$MODE" == "--build" ]]; then
    # Handle ./test.sh --build (no mode, just build flag)
    MODE="cpu"
    BUILD_FLAG="--build"
    IMAGE="visual-mic-dev"
    DOCKERFILE=""
    RUN_FLAGS=""
else
    IMAGE="visual-mic-dev"
    DOCKERFILE=""
    RUN_FLAGS=""
fi

# Build image if it doesn't exist or --build flag passed
if [[ "$BUILD_FLAG" == "--build" ]] || ! docker image inspect ${IMAGE} &>/dev/null; then
    echo "Building test image (${MODE})..."
    docker build \
        ${DOCKERFILE} \
        -t ${IMAGE} .
    echo ""
fi

echo "=== Lint ==="
docker run --rm --entrypoint "" ${IMAGE} ruff check .

echo ""
echo "=== Tests ==="
docker run --rm ${RUN_FLAGS} --entrypoint "" ${IMAGE} python -m pytest tests/ -v

echo ""
echo "All checks passed."
