#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-3.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-5.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-15.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-17.json --entry src/continuing_main.py --cpus 30
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-small-sweep/learning_curve.py save pdf
