#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

dirpath_recording=$1
dirpath_mistake=./repos/assembly101-mistake-detection/annots/
dirpath_coarse=./repos/assembly101-annotations/coarse-annotations/coarse_labels/
dirpath_fine=./repos/assembly101-annotations/fine-grained-annotations/
filepath_modification=data/annotation_modification.yaml
dirpath_output=./data/preprocess/
dirpath_log=./log/

python src/preprocess/format_annotation.py \
    --dirpath_recording "$dirpath_recording" \
    --dirpath_mistake "$dirpath_mistake" \
    --dirpath_coarse "$dirpath_coarse" \
    --dirpath_fine "$dirpath_fine" \
    --filepath_modification "$filepath_modification" \
    --dirpath_output "$dirpath_output" \
    --dirpath_log "$dirpath_log"
