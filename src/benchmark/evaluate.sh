#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate bridge

filepath_input=./output/prediction/$1
filepath_annotation=./data/preprocess/all_in_one.json
filepath_instruction=./data/graph/all.json
filepath_template=./src/benchmark/templates_eval.yaml
dirpath_output=./output/evaluation/
dirpath_log=./log

PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

model_id=gpt-4o-2024-11-20
template_type=ternary-step

python src/benchmark/evaluate.py \
    --filepath_input "$filepath_input" \
    --filepath_annotation "$filepath_annotation" \
    --filepath_instruction "$filepath_instruction" \
    --filepath_template "$filepath_template" \
    --dirpath_output "$dirpath_output" \
    --template_type "$template_type" \
    --model_id "$model_id" \
    --dirpath_log "$dirpath_log"
