#!/bin/sh

python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/DQN-11.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/Greedy-privileged.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/Greedy-hot.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/Greedy.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix/ForagerTemperature/Random.json --entry src/continuing_main.py --cpus 30


# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-temp-fix/learning_curve.py save pdf
