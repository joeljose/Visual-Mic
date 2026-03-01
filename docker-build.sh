#!/bin/bash
set -e

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg UNAME="$(whoami)" \
    -t visual-mic .

echo "Built visual-mic image as user: $(whoami) (uid=$(id -u), gid=$(id -g))"
