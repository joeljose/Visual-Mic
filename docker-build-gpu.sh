#!/bin/bash
set -e

VERSION=$(cat VERSION)

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    --build-arg VERSION="${VERSION}" \
    -f Dockerfile.gpu \
    -t visual-mic-gpu:${VERSION} \
    -t visual-mic-gpu:latest .

echo "Built visual-mic-gpu:${VERSION} (also tagged :latest)"
