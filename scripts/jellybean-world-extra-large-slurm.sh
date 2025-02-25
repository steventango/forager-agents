python scripts/slurm.py --cluster clusters/cedar_t3_c1_m2000_s4.json --runs 30 -e experiments/jellybean-world-extra-large/JellybeanWorldExtraLarge/Random.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/cedar_t8_c1_m2000_s1.json --runs 30 -e experiments/jellybean-world-extra-large/JellybeanWorldExtraLarge/DQN.json --entry src/continuing_main.py
