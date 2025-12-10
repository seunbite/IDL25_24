#!/bin/bash
source ~/.bashrc
conda activate km
cd ~/workspace/Career-Pathway

arg1=${1:-"English"}

python adhoc/require/make.py run2 --start_idx ${arg1}