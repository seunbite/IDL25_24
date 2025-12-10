#!/bin/bash
source ~/.bashrc
conda activate km
cd ~/workspace/Career-Pathway
echo $SLURM_NODELIST
export VLLM_ALLOW_LONG_MAX_MODEL_LEN='1'
export GPU_MEMORY_UTILIZATION='0.95'
arg1=${1:-"English"}
arg2=${2:-"h3_tree"}
arg3=${3:-"30,30,30,30"}
arg4=${4:-"False"}
arg5=${5:-"80"}

if [ "$arg1" = "h1_lexical" ]; then
    python adhoc/hypothesis/${arg1}.py
elif [ "$arg2" = "h4_gar" ]; then
    python adhoc/hypothesis/${arg2}.py --model_name_or_path ${arg1}
elif [ "$arg2" = "h5_rag_tree" ]; then
    python adhoc/hypothesis/${arg2}.py --model_name_or_path ${arg1} --threshold ${arg3}
elif [ "$arg2" = "h3_tree" ]; then
    python adhoc/hypothesis/${arg2}.py gentree --model_name_or_path ${arg1} --top_k_list ${arg3} --do_bias ${arg4} --beam_size ${arg5}
fi