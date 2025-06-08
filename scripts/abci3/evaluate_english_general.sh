#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0134_eval
#PBS -v RTYPE=rt_HG
#PBS -l select=1:ngpus=8
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/
#PBS -m n

set -eu -o pipefail

cd $PBS_O_WORKDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
source .venv_harness_en/bin/activate

GENERAL_TASK_NAME="triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en,squadv2"
GENERAL_NUM_FEWSHOT=4
GENERAL_NUM_TESTCASE="all"
GENERAL_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${GENERAL_NUM_FEWSHOT}shot_${GENERAL_NUM_TESTCASE}cases/general"

mkdir -p $GENERAL_OUTDIR

cd lm-evaluation-harness-en

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,max_length=4096,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $GENERAL_TASK_NAME \
    --num_fewshot $GENERAL_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "../$GENERAL_OUTDIR" \
    --use_cache "../$GENERAL_OUTDIR" \
    --log_samples \
    --seed 42

python scripts/aggregate_result.py --model $MODEL_NAME_PATH
