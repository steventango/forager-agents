#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/big/Forager/DQN.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/big/JellybeanWorld/DQN.json --entry src/continuing_main.py --cpus 30
python experiments/forager/learning_curve.py save pdf
