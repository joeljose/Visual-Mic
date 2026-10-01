#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg VERSION="${VERSION}" \
    -t visual-mic:${VERSION} \
    -t visual-mic:latest .

echo "Built visual-mic:${VERSION} (also tagged :latest)"
