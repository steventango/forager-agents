#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/big/Forager-1-percent-sweep/DQN.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/big/JellybeanWorld-1-percent-sweep/DQN.json --entry src/continuing_main.py --cpus 30
python experiments/big/learning_curve.py save pdf
