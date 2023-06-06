#!/usr/bin/env bash

DIR=$(realpath $(dirname "$0"))

FILES=$(find "$DIR" \
    -iname "*.md" \
    -o -iname "*.bib")

echo "$FILES" | entr -s "$DIR/build.sh"
