#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/jellybean-world-extra-large-sweep/JellybeanWorldExtraLarge/DQN.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 5 -e experiments/jellybean-world-extra-large-sweep/JellybeanWorldExtraLarge/SAC.json --entry src/continuing_main.py --cpus 30 --gpu
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/jellybean-world-extra-large-sweep/learning_curve.py save pdf
