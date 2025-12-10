#!/bin/bash
source ~/.bashrc
conda activate km

# Set default argument to "diversity" if no argument is provided
arg1=${1:-"diversity"}
arg2=${2:-"diversity"}
arg3=${3:-"start"}
arg4=${4:-"/scratch2/iyy1112/results/mcts_value_model/tmp_3_{}"}

# Fix if statement syntax
if [ "$arg1" = "truthfulness" ]; then
    python adhoc/eval/score_truthfulness.py --per_lang ${arg2} > "results/scores/${arg1}.txt"
elif [ "$arg1" = "riasec" ]; then
    python adhoc/eval/score_riasec.py eval_riasec --file_dir results/${arg2} > "results/scores/${arg1}_${arg2}.txt"
elif [ "$arg1" = "rag_d" ]; then
    python adhoc/eval/score_diversity.py scoring_diversity --file_name_format "results/baseline_retrieve/{}" > "results/scores/rag_${arg1}.txt"
elif [ "$arg1" = "rag_s" ]; then
    python adhoc/eval/score_diversity.py scoring_soundedness --file_name_format "results/baseline_retrieve/{}" > "results/scores/rag_${arg1}.txt"
elif [ "$arg1" = "diversity" ]; then
    python adhoc/eval/score_diversity.py scoring_diversity --file_name_format "results/eval_diversity/{}" > "${arg1}_gpt.txt"
elif [ "$arg1" = "diversity_40" ]; then
    python adhoc/eval/score_diversity.py scoring_diversity --file_name_format "results/eval_diversity_40/{}" > "${arg1}.txt"
elif [ "$arg1" = "soundedness" ]; then
    python adhoc/eval/score_diversity.py scoring_soundedness --file_name_format "results/eval_diversity/{}" > "results/scores/${arg1}_gpt.txt"
elif [ "$arg1" = "soundedness_40" ]; then
    python adhoc/eval/score_diversity.py scoring_soundedness --file_name_format "results/eval_diversity_40/{}" > "results/scores/${arg1}.txt"
elif [ "$arg1" = "soundedness_baseline" ]; then
    python adhoc/eval/get_baseline.py diversity > "${arg1}.txt"
elif [ "$arg1" = "diversity_mcts" ]; then
    python adhoc/mcts/score_sim_and_div.py --type ${arg2} --data_start ${arg3} --file_format ${arg4} > "results/scores/h_${arg2}_${arg3}_h5.txt"
elif [ "$arg1" = "value" ]; then
    python adhoc/eval/eval_value.py --file_name_format ${arg2} --start ${arg3} --value ${arg4}
    # python adhoc/mcts/score_sim_and_div.py --filenames "['results/1_lexical/tree_retrieval_semantic.jsonl']" --type 80 > mcts_h1.txt
fi
