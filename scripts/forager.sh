#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager/Forager-debug/DQN.json --entry src/continuing_main.py --cpus 30 # --gpu
# python scripts/local.py --runs 30 -e experiments/forager/Forager/DQN.json --entry src/continuing_main.py --cpus 30 --gpu
python experiments/forager/learning_curve.py save pdf
