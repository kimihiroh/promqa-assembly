#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

filepath_input=./data/preprocess/$1
filepath_all=./data/preprocess/all_in_one.json
dirpath_image=./data/parts/360p/
filepath_template=./src/qa-generation/templates.yaml
dirpath_output=./data/qa-generation/
dirpath_log=./log/

template_type="default"
model_id="gpt-4o-2024-11-20"
granularity="coarse+fine_all"

PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

python src/qa-generation/run.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"
