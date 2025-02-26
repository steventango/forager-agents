#!/bin/sh

# exit script on error
set -e

python scripts/slurm.py --cluster clusters/cedar_t1_c1_m2000_s1.json --runs 30 -e experiments/forager-extra-large/ForagerExtraLarge/SAC.json --entry src/continuing_main.py
