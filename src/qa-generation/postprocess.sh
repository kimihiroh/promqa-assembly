#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

filepath_input=./data/qa-generation/$1
filepath_all=./data/preprocess/all_in_one.json
dirpath_output=./data/qa-generation/
dirpath_log=./log/

PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

python src/qa-generation/postprocess.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_output "$dirpath_output" \
    --dirpath_log "$dirpath_log"
