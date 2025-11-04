#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate bridge

filepath_input=./data/all_v1.json
filepath_instruction=./data/graph/all.json
dirpath_instruction_image=./data/graph/raw/
dirpath_parts_image=./data/parts/360p/
dirpath_frame=$1/frames/rgb/360p/
filepath_template=./src/benchmark/templates.yaml
dirpath_output=./output/prediction/
dirpath_log=./log

model_id=$2
resolution=360p
max_frames=20
color=rgb
angle=C10115_rgb
mode=$3
reasoning_effort=${4:-"minimal"}


PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

python src/benchmark/inference_api.py \
    --filepath_input "$filepath_input" \
    --filepath_instruction "$filepath_instruction" \
    --dirpath_instruction_image "$dirpath_instruction_image" \
    --dirpath_parts_image "$dirpath_parts_image" \
    --dirpath_frame "$dirpath_frame" \
    --filepath_template "$filepath_template" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --resolution "$resolution" \
    --color "$color" \
    --angle "$angle" \
    --max_frames "$max_frames" \
    --mode "$mode" \
    --reasoning_effort "$reasoning_effort" \
    --dirpath_log "$dirpath_log"
