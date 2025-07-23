#!/bin/sh

# exit script on error
set -e

# python scripts/local.py --runs 1 -e experiments/forager-two-biome-small-video/ForagerTwoBiomeSmall/Random.json --entry src/continuing_main.py --cpus 1 &
# python scripts/local.py --runs 1 -e experiments/forager-two-biome-small-video/ForagerTwoBiomeSmall/Greedy.json --entry src/continuing_main.py --cpus 1 &
# python scripts/local.py --runs 1 -e experiments/forager-two-biome-small-video/ForagerTwoBiomeSmall/Greedy-122.json --entry src/continuing_main.py --cpus 1 &
# python scripts/local.py --runs 1 -e experiments/forager-two-biome-small-video/ForagerTwoBiomeSmall/DQN-15.json --entry src/continuing_main.py --cpus 1 &
# python scripts/local.py --runs 1 -e experiments/forager-two-biome-small-video/ForagerTwoBiomeSmall/DQN-5.json --entry src/continuing_main.py --cpus 1 &
# python scripts/local.py --runs 1 -e experiments/forager-two-biome-small-video/ForagerTwoBiomeSmall/DQN-5-mlp.json --entry src/continuing_main.py --cpus 1 &
# wait
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/Random.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/Greedy.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/Greedy-122.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 10 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DRQN-5.json --entry src/continuing_main.py --cpus 30 --gpu --record
# python scripts/local.py --runs 10 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DRQN-15.json --entry src/continuing_main.py --cpus 15 --gpu --record
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-3.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-5.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-5-mlp.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-15.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-15-mlp.json --entry src/continuing_main.py --cpus 30
# python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DQN-17.json --entry src/continuing_main.py --cpus 30
python experiments/forager-two-biome-small/learning_curve.py save pdf
