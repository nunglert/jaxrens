#!/usr/bin/env bash
# Canonical "does jaxrens build and run?" check sequence.
# Usage: scripts/ci-checks.sh [smoke|full]
#   smoke (default) — fast tier, excludes heavy markers
#   full            — every test, including heavy
set -euo pipefail

TIER="${1:-smoke}"

echo "=== [1/4] Import smoke ==="
python -c "import jaxrens; print('jaxrens at', jaxrens.__file__)"

echo "=== [2/4] CLI entry point ==="
jaxrens --help > /dev/null
jaxrens dump-schema > /dev/null

echo "=== [3/4] Example config validation ==="
jaxrens validate -c experiments/examples/lj8_npt/config.yaml
jaxrens validate -c experiments/examples/l64_batch/config.yaml

echo "=== [4/4] Pytest (tier=$TIER) ==="
case "$TIER" in
    smoke)
        # Default markers (excludes 'heavy' per pyproject); keep 'gpu' since
        # we run with --gpus all. Skip 'multi_gpu' — single-GPU box.
        pytest tests/ -x -q -m "not multi_gpu"
        ;;
    full)
        # Everything except multi-GPU.
        pytest tests/ -v -m "not multi_gpu" -o addopts=""
        ;;
    *)
        echo "Unknown tier: $TIER (expected: smoke | full)" >&2
        exit 2
        ;;
esac
