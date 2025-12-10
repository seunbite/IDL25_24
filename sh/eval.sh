#!/bin/bash
source ~/.bashrc
conda activate km
cd ~/workspace/Career-Pathway
export VLLM_ALLOW_LONG_MAX_MODEL_LEN='1'
export GPU_MEMORY_UTILIZATION='0.95'
arg1=${1:-"English"}
arg2=${2:-"English"}
arg3=${3:-"us"}
arg4=${4:-"10"}

if [ "$arg2" = "diversity" ]; then
    cmd="python adhoc/eval/eval_${arg2}.py --model_name_or_path ${arg1}"
elif [ "$arg2" = "truthfulness" ]; then
    cmd="python adhoc/eval/eval_${arg2}.py --model_name_or_path ${arg1}"
elif [ "$arg2" = "bias" ]; then
    cmd="python adhoc/eval/eval_${arg2}.py --model_name_or_path ${arg1} --start ${arg3}"
elif [ "$arg2" = "issue" ]; then
    cmd="python adhoc/eval/eval_${arg2}.py main --model_name_or_path ${arg1}"
elif [ "$arg2" = "issue_annot" ]; then
    cmd="python adhoc/eval/eval_issue.py annotate_data --type ${arg1} --start ${arg3}"
elif [ "$arg2" = "issue_data" ]; then
    cmd="python adhoc/eval/eval_issue.py datacuration"
elif [ "$arg2" = "bias_name" ]; then
    cmd="python adhoc/eval/gdp_and_name.py"
fi

# cmd="python adhoc/eval/score_riasec.py --male ${arg1}"
echo $cmd
eval $cmd

