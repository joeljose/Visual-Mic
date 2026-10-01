#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg VERSION="${VERSION}" \
    -f Dockerfile.gpu \
    -t visual-mic-gpu:${VERSION} \
    -t visual-mic-gpu:latest .

echo "Built visual-mic-gpu:${VERSION} (also tagged :latest)"
