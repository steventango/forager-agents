#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/forager-extra-large/ForagerExtraLarge/Greedy.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 1 -e experiments/forager-extra-large/ForagerExtraLarge/Random.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-extra-large/ForagerExtraLarge/DQN.json --entry src/continuing_main.py --cpus 30
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-extra-large/learning_curve.py save pdf
