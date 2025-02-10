#!/bin/sh

# exit script on error
set -e

python src/continuing_main.py -e experiments/debug/Forager-1-percent-sweep/DQN.json -i 0
python src/continuing_main.py -e experiments/debug/JellybeanWorld-1-percent-sweep/DQN.json -i 0
