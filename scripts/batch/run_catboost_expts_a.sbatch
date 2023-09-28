#!/bin/bash -x

#SBATCH --output=run_catboost_expts_a-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=99:59:00
#SBATCH --mem=32GB
#SBATCH --job-name=run_catboost_expts_a
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

srun \
    cd /scratch/bf996/tabzilla/scripts/experiments/; \
    ls; \
    cd experiments; \
    ls; \
    source algs_catboost_only_a.sh; \
    echo "Tabzilla experiments done."; \
