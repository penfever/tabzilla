#!/bin/bash
set -e

# load functions
source ../utils.sh

##############################
# begin: EXPERIMENT PARAMETERS

# this defines MODELS_ENVS
source ../catboost_tabpfn.sh

# this defines DATASETS
source ../LARGE_DATASETS_BENCHMARK.sh

# this defines CONFIG_FILE_NAMES
source ../config_settings.sh

name=algs-catboost-tabpfn-large

# base name for the gcloud instances
instance_base=$name

# experiment name (will be appended to results files)
experiment_name=$name

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# counter for total number of jobs launched
export CUR_JOB_COUNT=0

# results file: check for results here before launching each experiment
result_log=/scratch/bf996/tabzilla/TabZilla/result_log.txt

# conda envs
TORCH_ENV="torch"
KERAS_ENV="tensorflow"
SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"

# conda scripts
TORCH_BASH="run_tabzilla_env.bash"
GBDT_BASH="run_tabzilla_env_gbdt.bash"

# end: EXPERIMENT PARAMETERS
############################

####################
# begin: bookkeeping

# make a log directory
mkdir -p ${PWD}/logs
LOG_DIR=${PWD}/logs

#################
# run experiments

num_experiments=0
for h in ${!CONFIG_FILE_NAMES[@]};
do
target_config="${CONFIG_FILE_NAMES[h]}"
  for i in ${!MODELS_ENVS[@]};
  do
    for j in ${!DATASETS[@]};
    do
      model_env="${MODELS_ENVS[i]}"
      catboost_env="CatBoost:gbdt"
      tabpfn_env="TabPFNModel:torch"
      if [ "$model_env" = "$catboost_env" ]; then
          bash_script=$GBDT_BASH
          gpu_text="SBATCH --gres=gpu:1"
      else
          bash_script=$TORCH_BASH
          gpu_text="SBATCH --gres=gpu:1"
      fi

      model="${model_env%%:*}"
      env="${model_env##*:}"

      # if the experiment is already in the result log, skip it
      if grep -Fxq "${DATASETS[j]},${model},${experiment_name}" ${result_log}; then
        echo "experiment found in logs. skipping. dataset=${DATASETS[j]}, model=${model}, expt=${experiment_name}"
        continue
      fi

      instance_name=${instance_base}-${i}-${j}

      # args:
      # $1 = model name
      # $2 = dataset name
      # $3 = env name
      # $4 = instance name
      # $5 = experiment name
      echo "MODEL_ENV: ${model_env}"
      echo "MODEL: ${model}"
      echo "ENV: ${env}"
      echo "DATASET: ${DATASETS[j]}"
      echo "EXPERIMENT_NAME: ${experiment_name}"

      config_file=/scratch/bf996/tabzilla/TabZilla/tabzilla_experiment_config_gpu_mod_${i}_${j}.yml
      result_name=${DATASETS[j]}-${model}-${i}-${j}
      sed -e "s#%%RESULTS_NAME%%#${result_name}#g" \
      ${target_config} > ${config_file}
      sleep 1
      sed -e "s#%%EXPT_CONFIG%%#${config_file}#g" \
      -e "s#%%GPU_CONFIG%%#${gpu_text}#g" \
      -e "s#%%ENV_CONFIG%%#${bash_script}#g" \
      -e "s#%%DATASET_PATH%%#${DATASETS[j]}#g" \
      -e "s#%%MODEL_NAME%%#${model}#g" \
      /scratch/bf996/tabzilla/scripts/batch/tabzilla_batch.sbatch > /scratch/bf996/tabzilla/scripts/batch/tabzilla_batch_mod.sbatch
      sleep 1
      run_experiment_slurm "${model}" ${DATASETS[j]} ${env} ${instance_base}-${i}-${j} ${experiment_name} ${config_file} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
      num_experiments=$((num_experiments + 1))

      # add instance name to the instance list
      INSTANCE_LIST+=("${instance_name}")

      echo "launched instance ${instance_base}-${i}-${j}. (job number ${num_experiments})"
      sleep 2

      # if we have started MAX_PROCESSES experiments, wait for them to finish
      while true; do
      # Check if the output of the squeue command is empty
      cur_job_count_squeue=$(squeue -u $(whoami) | tail -n +2 | wc -l)
      if [ "$cur_job_count_squeue" -ge "$MAX_PROCESSES" ]; then
          sleep 120
      else
          break
      fi
      done
    done
  done
done

echo "still waiting for processes to finish..."
wait
echo "done."