#!/bin/bash
source ~/.bashrc
conda activate machi
cd /home/iyy1112/workspace/Career-Pathway

# python scrap/translate.py
# python scrap/annotate.py
python datacuration/urls_to_video2dataset.py