python scripts/local.py --runs 10 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DRQN-5.json --entry src/continuing_main.py --cpus 30 --gpu --record
python experiments/forager-two-biome-small/learning_curve.py save pdf
bash scripts/forager-two-biome-small-sweep.sh

python scripts/local.py --runs 10 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DRQN-15.json --entry src/continuing_main.py --cpus 15 --gpu --record
python experiments/forager-two-biome-small/learning_curve.py save pdf

python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DRQN-5.json --entry src/continuing_main.py --cpus 30 --gpu --record
python scripts/local.py --runs 30 -e experiments/forager-two-biome-small/ForagerTwoBiomeSmall/DRQN-15.json --entry src/continuing_main.py --cpus 15 --gpu --record
python experiments/forager-two-biome-small/learning_curve.py save pdf
