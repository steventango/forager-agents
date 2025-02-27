#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-temperature-sweep/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 30
NUMBA_DISABLE_JIT=1 python experiments/forager-temperature-sweep/learning_curve.py save pdf
