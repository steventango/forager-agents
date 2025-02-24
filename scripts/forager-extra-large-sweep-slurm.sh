#!/bin/sh

# exit script on error
set -e

python scripts/slurm.py --cluster clusters/debug_cedar.json --runs 1 -e experiments/forager-extra-large-sweep/ForagerExtraLarge/Random.json --entry src/continuing_main.py
