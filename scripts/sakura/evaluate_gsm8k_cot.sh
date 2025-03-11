#!/bin/bash
#SBATCH --job-name=0107
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -eu -o pipefail

source .venv_harness_en/bin/activate

# This script is used to evaluate
# triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en
# to evaluate with all testcases, set NUM_TESTCASE=None

MODEL_NAME_PATH=$1
TP=1
DP=1

GSM8K_TASK_NAME="gsm8k_cot,gsm8k"
GSM8K_NUM_FEWSHOT=8
GSM8K_NUM_TESTCASE="gsm8k"
GSM8K_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${GSM8K_NUM_FEWSHOT}shot_${GSM8K_NUM_TESTCASE}cases/gsm8k"

mkdir -p $GSM8K_OUTDIR

cd lm-evaluation-harness-en

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=$DP \
    --tasks $GSM8K_TASK_NAME \
    --num_fewshot $GSM8K_NUM_FEWSHOT \
    --batch_size auto \
    --device cuda \
    --write_out \
    --output_path "../$GSM8K_OUTDIR" \
    --use_cache "../$GSM8K_OUTDIR" \
    --log_samples \
    --seed 42 \

