#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-3.json --entry src/continuing_main.py --cpus 30 # --gpu
python scripts/local.py --runs 30 -e experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-5.json --entry src/continuing_main.py --cpus 30 # --gpu
python scripts/local.py --runs 30 -e experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-7.json --entry src/continuing_main.py --cpus 30 # --gpu
python scripts/local.py --runs 30 -e experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-9.json --entry src/continuing_main.py --cpus 30 # --gpu
python scripts/local.py --runs 30 -e experiments/forager-two-biome-large/ForagerTwoBiomeLarge/DQN-15.json --entry src/continuing_main.py --cpus 30 # --gpu
python experiments/forager-two-biome-large/learning_curve.py save pdf
