"""
Predict answers from each set of a video (images), a recipe, and a question.

"""

from argparse import ArgumentParser
from copy import deepcopy
import json
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

from vllm import LLM, SamplingParams
import torch

from utils import (
    format_instruction,
)

from src.utils.misc import get_date


def load_model(args):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    model = LLM(
        model=args.model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=torch.bfloat16,
        seed=args.seed,
        trust_remote_code=True,
    )

    return model, sampling_params


def format_input(args, examples, template_components, toy2instruction):
    logging.info("Prepare input ... ")

    inputs = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        prompt = ""

        # prefix
        prompt += template_components["prefix"] + "\n"

        # assembly manual
        prompt += (
            template_components["manual"]["dot"].replace(
                "{dot}", toy2instruction[example["toy_id"]]["dot"]
            )
            + "\n"
        )

        # note for text-only
        prompt += template_components["note"]["text-only"] + "\n"

        # task
        prompt += template_components["task-text"].replace(
            "{question}", example["question"]
        )

        prompt += "<think>\n"

        inputs.append(prompt)

    return inputs


def predict(model, sampling_params, inputs):
    logging.info("Prediction starts ... ")
    _outputs = model.generate(inputs, sampling_params)
    outputs = [x.outputs[0].text for x in _outputs]
    return outputs


def save(args, examples, inputs, predictions):
    new_examples = []
    for example, _input, prediction in zip(examples, inputs, predictions):
        new_example = deepcopy(example)
        new_example["prediction"] = {
            "text_prompt": _input,
            "response": prediction,
        }
        new_examples.append(new_example)

    filepath_output = (
        args.dirpath_output
        / f"text-only_{Path(args.model_id).name}.json"
    )

    output = {
        "metadata": {
            "data-created": get_date(),
            "model_id": args.model_id,
            "input": args.filepath_input.name,
        },
        "examples": new_examples,
    }

    with open(filepath_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


def main(args):
    examples = []
    for toy_id in SUBSETS:
        examples += load_dataset("kimihiroh/promqa-assembly", toy_id, split='test')

    toy2instruction = format_instruction()

    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    inputs = format_input(
        args,
        examples,
        template_components,
        toy2instruction,
    )

    logging.info("Sanity Check ...")
    idx = 0
    logging.info(f"{idx=}")
    logging.info(inputs[idx])

    model, sampling_params = load_model(args)

    predictions = predict(model, sampling_params, inputs)

    save(args, examples, inputs, predictions)


if __name__ == "__main__":
    parser = ArgumentParser(description="Predict")
    
    parser.add_argument("--model_id", type=str, help="model id")
    
    # parser.add_argument("--filepath_input", type=Path, help="filepath for input")
    # parser.add_argument(
    #     "--filepath_instruction", type=Path, help="filepath for instruction"
    # )
    parser.add_argument(
        "--filepath_template", type=Path, help="filepath for prompt template",
        default='.src/benchmark/template.yaml'
    )
    parser.add_argument(
        "--dirpath_output", type=Path, help="filepath for output",
        default='./output/prediction'
    )
    
    parser.add_argument("--temperature", type=float, help="temperature", default=0.6)
    parser.add_argument("--top_p", type=float, help="top p", default=1)
    parser.add_argument(
        "--max_tokens", type=int, help="max tokens to generate", default=2048
    )
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / f"text-only_{get_date()}.log"),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
