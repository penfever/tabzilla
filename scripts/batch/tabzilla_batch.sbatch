#!/bin/bash -x

#SBATCH --output=run_tabzilla_expt-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_tabzilla_expt
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MASTER_PORT=$(shuf -i 10000-65500 -n 1);
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu";

srun \
    /bin/bash /scratch/bf996/tabzilla/scripts/%%ENV_CONFIG%% \
    /bin/bash -c \
    'python tabzilla_experiment.py --experiment_config %%EXPT_CONFIG%% --dataset_dir ./datasets/%%DATASET_PATH%% --model_name %%MODEL_NAME%%'