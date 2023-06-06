#!/usr/bin/env bash

DIR=$(realpath $(dirname "$0"))

docker run --rm \
    --volume $DIR:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
