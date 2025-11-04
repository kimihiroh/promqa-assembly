#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

filepath_input=./data/preprocess/all_in_one.json
filepath_graph=./data/preprocess/graph/all.json
filepath_output=./data/preprocess/all_examples.json
dirpath_log=./log

python src/preprocess/create_example.py \
    --filepath_input "$filepath_input" \
    --filepath_graph "$filepath_graph" \
    --filepath_output "$filepath_output" \
    --dirpath_log "$dirpath_log"
