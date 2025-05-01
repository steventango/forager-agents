#!/bin/sh

python scripts/local.py --runs 1 -e experiments/forager-temp-fix-slow-video/ForagerTemperature/Random.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/forager-temp-fix-slow-video/ForagerTemperature/Greedy.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/forager-temp-fix-slow-video/ForagerTemperature/DQN-9.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/forager-temp-fix-slow-video/ForagerTemperature/DQN-privileged.json --entry src/continuing_main.py --cpus 1 & 
python scripts/local.py --runs 1 -e experiments/forager-temp-fix-slow-video/ForagerTemperature/Greedy-privileged.json --entry src/continuing_main.py --cpus 1
