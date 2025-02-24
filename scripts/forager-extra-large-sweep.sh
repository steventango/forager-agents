#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-extra-large-sweep/ForagerExtraLarge/DQN.json --entry src/continuing_main.py --cpus 30
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-extra-large-sweep/learning_curve.py save pdf
