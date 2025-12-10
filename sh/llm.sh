#!/bin/bash
source ~/.bashrc
conda activate machi
cd ~/workspace/Career-Pathway
export VLLM_ALLOW_LONG_MAX_MODEL_LEN='1'

arg1=${1:-"English"}

python adhoc/require/ordering.py --start ${arg1}