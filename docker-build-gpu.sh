#!/bin/bash
set -e

docker build \
    --network=host \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -f Dockerfile.gpu \
    -t visual-mic-gpu .

echo "Built visual-mic-gpu image as user: $(whoami) (uid=$(id -u), gid=$(id -g))"
