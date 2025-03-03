#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 1 -e experiments/forager-temperature/ForagerTemperature/Greedy.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/forager-temperature/ForagerTemperature/Random.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/forager-temperature/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/forager-temperature/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 1 &
wait
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-temperature/learning_curve.py save pdf
