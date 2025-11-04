#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate bridge

filepath_input=./data/preprocess/$1
filepath_graph=./data/graph/all.json
dirpath_output=./data/graph/status/
dirpath_log=./log/

python src/graph-annotation/create_image_w_status.py \
    --filepath_input "$filepath_input" \
    --filepath_graph $filepath_graph \
    --dirpath_output $dirpath_output \
    --dirpath_log $dirpath_log
