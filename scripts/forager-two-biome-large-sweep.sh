#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-3.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-5.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-7.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-9.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-11.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-13.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-15.json --entry src/continuing_main.py --cpus 30
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-large-sweep/learning_curve.py save pdf
