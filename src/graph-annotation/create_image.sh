#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate bridge

filepath_input=./data/graph/all.json
dirpath_parts=./data/parts/
dirpath_output=./data/graph/raw
dirpath_log=./log/

python src/graph-annotation/create_image.py \
    --filepath_input "$filepath_input" \
    --dirpath_parts "$dirpath_parts" \
    --dirpath_output "$dirpath_output" \
    --dirpath_log $dirpath_log
