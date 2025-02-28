#!/bin/sh

<<<<<<< HEAD
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-11.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Greedy-privileged.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Greedy-hot.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Greedy.json --entry src/continuing_main.py --cpus 30
python scripts/local.py --runs 30 -e experiments/forager-temp-fix-slow/ForagerTemperature/Random.json --entry src/continuing_main.py --cpus 30

=======
# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-11.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 5 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 5 -e experiments/forager-temp-fix-slow/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 5 &
wait
>>>>>>> 9e1c76b3210a71afd9392340e8fee27e4e404ffb
# ReferenceError: underlying object has vanished, caused by NumPyRandomGeneratorType
NUMBA_DISABLE_JIT=1 python experiments/forager-temp-fix-slow/learning_curve.py save pdf
