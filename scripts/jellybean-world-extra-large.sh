#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 30 -e experiments/jellybean-world-extra-large/JellybeanWorldExtraLarge/Greedy.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/jellybean-world-extra-large/JellybeanWorldExtraLarge/DQN.json --entry src/continuing_main.py --cpus 30
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/jellybean-world-extra-large/learning_curve.py save pdf
