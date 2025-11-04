#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate promqa-assembly

filepath_input=./data/preprocess/$1
filepath_all=./data/preprocess/all_in_one.json
dirpath_image=./parts/360p/
filepath_template=./src/qa-generation/templates.yaml
dirpath_output=./data/qa-generation/prompt-engineering/
dirpath_log=./log/

##################### context #####################
template_type="default"
model_id="gpt-4o-2024-11-20"

granularity="coarse"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"

granularity="coarse+fine_all"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### model #####################
template_type="default"
granularity="coarse"

model_id="claude-3-7-sonnet-20250219"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"

model_id="gemini-2.5-pro-exp-03-25"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"

model_id="o3-mini-2025-01-31"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### parts image #####################
model_id="gpt-4o-2024-11-20"
granularity="coarse"

template_type="text+image"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### no demonstration #####################
template_type="no-demonstration"
model_id="gpt-4o-2024-11-20"

granularity="coarse"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### multi runs #####################
template_type="question"
model_id="gpt-4o-2024-11-20"

granularity="coarse"
filepath_input=./data/promqa/assembly101/preprocess/$1
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


template_type="answer"
model_id="gpt-4o-2024-11-20"

granularity="coarse"
filepath_input=./data/promqa/assembly101/generation/prompt-engineering/samples_50.question.gpt-4o-2024-11-20.coarse.False.json
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### multi runs w/ images #####################
template_type="question+image"
model_id="gpt-4o-2024-11-20"

filepath_input=./data/promqa/assembly101/preprocess/$1
granularity="coarse"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


template_type="answer+image"
model_id="gpt-4o-2024-11-20"

granularity="coarse"
filepath_input=./data/promqa/assembly101/generation/prompt-engineering/samples_50.question+image.gpt-4o-2024-11-20.coarse.False.json
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### multi runs w/ images & fine #####################
template_type="question+image"
model_id="gpt-4o-2024-11-20"

filepath_input=./data/promqa/assembly101/preprocess/$1
granularity="coarse+fine_all"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


template_type="answer+image"
model_id="gpt-4o-2024-11-20"

granularity="coarse+fine_all"
filepath_input=./data/promqa/assembly101/generation/prompt-engineering/samples_50.question+image.gpt-4o-2024-11-20.coarse+fine_all.False.json
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"


##################### bias analysis #####################
template_type="default"
granularity="coarse+fine_all"

model_id="gpt-4o-2024-11-20"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"

model_id="claude-3-7-sonnet-20250219"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"

model_id="gemini-2.0-flash-001"
python src/generation/run_a101.py \
    --filepath_input "$filepath_input" \
    --filepath_all "$filepath_all" \
    --dirpath_image "$dirpath_image" \
    --dirpath_output "$dirpath_output" \
    --model_id "$model_id" \
    --template_type "$template_type" \
    --filepath_template "$filepath_template" \
    --granularity "$granularity" \
    --dirpath_log "$dirpath_log"

##################### below is skipped #####################
###################################################

# granularity="coarse+fine_part"
# python src/generation/run_a101.py \
#     --filepath_input "$filepath_input" \
#     --filepath_all "$filepath_all" \
#     --dirpath_image "$dirpath_image" \
#     --dirpath_output "$dirpath_output" \
#     --model_id "$model_id" \
#     --template_type "$template_type" \
#     --filepath_template "$filepath_template" \
#     --granularity "$granularity" \
#     --dirpath_log "$dirpath_log"

##################### screw #####################
# template_type="default"
# model_id="gpt-4o-2024-11-20"
# granularity="coarse"

# # w/ screw
# python src/generation/run_a101.py \
#     --filepath_input "$filepath_input" \
#     --filepath_all "$filepath_all" \
#     --dirpath_image "$dirpath_image" \
#     --dirpath_output "$dirpath_output" \
#     --model_id "$model_id" \
#     --template_type "$template_type" \
#     --filepath_template "$filepath_template" \
#     --granularity "$granularity" \
#     --w_screw \
#     --dirpath_log "$dirpath_log"


##################### inline demonstration #####################
# template_type="demonstration"
# model_id="gpt-4o-2024-11-20"

# granularity="coarse"
# python src/generation/run_a101.py \
#     --filepath_input "$filepath_input" \
#     --filepath_all "$filepath_all" \
#     --dirpath_image "$dirpath_image" \
#     --dirpath_output "$dirpath_output" \
#     --model_id "$model_id" \
#     --template_type "$template_type" \
#     --filepath_template "$filepath_template" \
#     --granularity "$granularity" \
#     --dirpath_log "$dirpath_log"
