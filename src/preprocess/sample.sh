#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

filepath_input=./data/preprocess/all_examples.json
dirpath_output=./data/preprocess
dirpath_log=./log

num_total_sample=$1

python src/preprocess/sample.py \
    --filepath_input "$filepath_input" \
    --dirpath_output "$dirpath_output" \
    --num_total_sample "$num_total_sample" \
    --dirpath_log "$dirpath_log"
