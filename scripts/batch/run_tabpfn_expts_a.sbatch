#!/bin/bash -x

#SBATCH --output=run_tabpfn_all_expts-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=99:59:00
#SBATCH --mem=128GB
#SBATCH --job-name=run_tabpfn_all_expts
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

srun \
    cd /scratch/bf996/tabzilla/scripts/experiments/; \
    ls; \
    cd experiments; \
    ls; \
    source algs_tabpfn_only_a.sh; \
    echo "Tabzilla experiments done."; \
