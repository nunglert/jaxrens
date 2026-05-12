#!/bin/bash
#SBATCH --job-name=jaxrens_bench
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# Repo root, whether sbatch was run from there or from bench/.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -d bench ]] || cd ..

mkdir -p bench/logs/tmp
exec >"bench/logs/slurm-${SLURM_JOB_ID:-local}.out" 2>&1
export TMPDIR="$PWD/bench/logs/tmp"

source ~/miniconda3/bin/activate jaxrens
python bench/run.py "$@"
