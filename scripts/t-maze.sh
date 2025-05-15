#!/bin/bash
set -e

python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DQN.json &
python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DRQN-1-32.json &
python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DRQN-2-16.json &
python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DRQN-4-8.json &
python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DRQN-8-4.json &
python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DRQN-16-2.json &
python scripts/local.py --runs 30 -e experiments/t-maze/TMaze/DRQN-32-1.json &
wait
python experiments/t-maze/learning_curve.py save pdf
