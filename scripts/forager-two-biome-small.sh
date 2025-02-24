#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/Random.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-3.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-17.json --entry src/continuing_main.py --cpus 30
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-small/learning_curve.py save pdf
