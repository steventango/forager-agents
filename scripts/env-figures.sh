
python scripts/local.py --runs 1 -e experiments/env-figures/ForagerExtraLarge/DQN.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/env-figures/ForagerTemperature/DQN.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/env-figures/ForagerTwoBiomeLarge/DQN-7.json --entry src/continuing_main.py --cpus 1 &
python scripts/local.py --runs 1 -e experiments/env-figures/ForagerTwoBiomeSmall/DQN-3.json --entry src/continuing_main.py --cpus 1 &
wait
