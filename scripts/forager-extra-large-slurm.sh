#!/bin/sh

# exit script on error
set -e

python scripts/slurm_gpu.py --cluster clusters/cedar_g_t12_c1_m2000_s1.json --runs 5 -e experiments/forager-extra-large/ForagerExtraLarge/SAC.json --entry src/continuing_main.py
