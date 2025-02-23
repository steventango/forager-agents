#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/jellybean-world-extra-large/JellybeanWorldExtraLarge/DQN.json --entry src/continuing_main.py --cpus 30
python experiments/jellybean-world-extra-large/learning_curve.py
