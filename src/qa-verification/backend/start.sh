#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

cd ./src/qa-verification/backend/ || exit

dirpath_input=$1
dirpath_output=$2

num_worker=1
host=0.0.0.0
port=5350

DIRPATH_INPUT="$dirpath_input" DIRPATH_OUTPUT="$dirpath_output" gunicorn -w "$num_worker" -b "$host":"$port" app:app
