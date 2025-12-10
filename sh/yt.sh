#!/bin/bash
source ~/.bashrc
conda activate machi
cd ~/workspace/Career-Pathway
export VLLM_ALLOW_LONG_MAX_MODEL_LEN='1'

arg1=${1:-"English"}
arg2=${2:-"English"}

python adhoc/yt_processing/yt_process.py --stage ${arg1} --start ${arg2}