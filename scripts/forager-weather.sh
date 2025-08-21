#!/bin/sh

python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/Greedy-privileged.json --entry src/continuing_main.py --cpus 10
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/Greedy.json --entry src/continuing_main.py --cpus 10
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/Random.json --entry src/continuing_main.py --cpus 10
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/Temperature.json --entry src/continuing_main.py --cpus 10

python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/DQN-9.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/DQN-9-memory.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/DRQN-9.json --entry src/continuing_main.py --cpus 5 &
python scripts/local.py --runs 10 -e experiments/forager-weather/ForagerTemperature/W0-DQN.json --entry src/continuing_main.py --cpus 5 &
wait

python experiments/forager-weather/learning_curve.py save pdf
