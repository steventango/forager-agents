#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DRQN-5.json --entry src/continuing_main.py --cpus 30 --gpu
python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DRQN-15.json --entry src/continuing_main.py --cpus 15 --gpu
# python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-3.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-5.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-15.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 5 -e experiments/forager-two-biome-small-sweep/ForagerTwoBiomeSmall/DQN-17.json --entry src/continuing_main.py --cpus 30
python experiments/forager-two-biome-small-sweep/learning_curve.py save pdf
python experiments/forager-two-biome-small-sweep/process_sweep.py
