#!/bin/sh

python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-5.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-7.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-9.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-11.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Greedy-privileged.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Greedy-hot.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Greedy.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Random.json --entry src/continuing_main.py --cpus 30

# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-temp-fix-slow/learning_curve.py save pdf
NUMBA_DISABLE_JIT=1 python experiments/forager-temp-fix-slow-long/learning_curve.py save pdf
