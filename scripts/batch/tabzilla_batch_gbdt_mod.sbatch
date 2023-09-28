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
    /bin/bash /scratch/bf996/tabzilla/scripts/run_tabzilla_env_gbdt.bash \
    /bin/bash -c \
    'python tabzilla_experiment.py --experiment_config /scratch/bf996/tabzilla/TabZilla/tabzilla_experiment_config_gpu_mod_0_17.yml --dataset_dir ./datasets/openml__splice__45 --model_name CatBoost'