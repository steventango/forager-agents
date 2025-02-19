#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-3.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/forager-two-biome-large-sweep/ForagerTwoBiomeLarge/DQN-15.json --entry src/continuing_main.py --cpus 30
python experiments/forager-two-biome-large-sweep/learning_curve.py save pdf
