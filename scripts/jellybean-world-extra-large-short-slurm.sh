python scripts/slurm_gpu.py --cluster clusters/cedar_g_t1_c1_m2000_s1.json --runs 1 -e experiments/jellybean-world-extra-large-short/JellybeanWorldExtraLarge/SAC.json --entry src/continuing_main.py
python scripts/slurm_gpu.py --cluster clusters/cedar_g_t1_c2_m2000_s1.json --runs 2 -e experiments/jellybean-world-extra-large-short/JellybeanWorldExtraLarge/SAC.json --entry src/continuing_main.py
