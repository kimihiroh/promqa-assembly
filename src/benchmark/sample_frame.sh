#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

# preprocess videos
dirpath_input=$1
filepath_annotation=./data/preprocess/all_in_one.json
dirpath_output=$2
dirpath_log=./log/

mkdir -p $dirpath_log

max_parallel_jobs=12

# note:
# sample frames from all recordings as most of them are used in QA
# QA example-based sampling would produce more duplicates.

echo "Start video resizing: $(date)"
python src/benchmark/sample_frame.py \
    --dirpath_input "$dirpath_input" \
    --filepath_annotation "$filepath_annotation" \
    --dirpath_output "$dirpath_output" \
    --max_parallel_jobs $max_parallel_jobs \
    --dirpath_log $dirpath_log
echo "Finish video resizing: $(date)"
