#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/setup_cuda_env.sh"

cd "$SCRIPT_DIR/.."

make clean
make all
