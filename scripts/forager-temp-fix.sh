#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-temp-fix/ForagerTemperature/DQN-11.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 5 -e experiments/forager-temp-fix/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 5 -e experiments/forager-temp-fix/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 5 &
wait
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-temp-fix/learning_curve.py save pdf
