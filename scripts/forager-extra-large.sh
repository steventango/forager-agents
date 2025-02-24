#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/forager-extra-large/ForagerExtraLarge/DQN.json --entry src/continuing_main.py --cpus 30
python experiments/forager-extra-large/learning_curve.py save pdf
