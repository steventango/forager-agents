#!/bin/sh

# exit script on error
set -e

python scripts/slurm.py --cluster clusters/cedar_g_t1_c3_m2000_s1.json --runs 1 -e experiments/forager-extra-large/ForagerExtraLarge/SAC.json --entry src/continuing_main.py
python scripts/slurm.py --cluster clusters/cedar_g_t3_c3_m2000_s1.json --runs 10 -e experiments/forager-extra-large/ForagerExtraLarge/SAC.json --entry src/continuing_main.py
