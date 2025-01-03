#!/bin/bash
#SBATCH --job-name=0107
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

source .venv_harness_en/bin/activate

# This script is used to evaluate
# triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en
# to evaluate with all testcases, set NUM_TESTCASE=None

MODEL_NAME_PATH=$1
TP=$2
DP=$3

GENERAL_TASK_NAME="triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en"
GENERAL_NUM_FEWSHOT=4
GENERAL_NUM_TESTCASE="all"
GENERAL_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${GENERAL_NUM_FEWSHOT}shot_${GENERAL_NUM_TESTCASE}cases/general"

MMLU_TASK_NAME="mmlu"
MMLU_NUM_FEWSHOT=5
MMLU_NUM_TESTCASE="all"
MMLU_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${MMLU_NUM_FEWSHOT}shot_${MMLU_NUM_TESTCASE}cases/mmlu"

BBH_TASK_NAME="bbh_cot_fewshot"
BBH_NUM_FEWSHOT=3
BBH_NUM_TESTCASE="all"
BBH_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${BBH_NUM_FEWSHOT}shot_${BBH_NUM_TESTCASE}cases/bbh_cot"

mkdir -p $GENERAL_OUTDIR
mkdir -p $MMLU_OUTDIR
mkdir -p $BBH_OUTDIR

cd lm-evaluation-harness-en

echo $MMLU_TASK_NAME
lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $MMLU_TASK_NAME \
    --num_fewshot $MMLU_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "../$MMLU_OUTDIR" \
    --use_cache "../$MMLU_OUTDIR" \
    --seed 42

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $BBH_TASK_NAME \
    --num_fewshot $BBH_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "../$BBH_OUTDIR" \
    --use_cache "../$BBH_OUTDIR" \
    --log_samples \
    --seed 42

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $GENERAL_TASK_NAME \
    --num_fewshot $GENERAL_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "../$GENERAL_OUTDIR" \
    --use_cache "../$GENERAL_OUTDIR" \
    --log_samples \
    --seed 42 \

