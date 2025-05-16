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
from PIL import Image

from utils import (
    format_instruction,
    sample_frame,
)

from src.utils.misc import get_date


def load_model(args):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["\n"],
    )
    # load model
    if args.model_id == "OpenGVLab/InternVL3-8B":
        model = LLM(
            model=args.model_id,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": args.max_frame + 1, "video": 0},  # parts+frames
            dtype=torch.bfloat16,
            seed=args.seed,
            max_model_len=32768,  # kh: max position embedding, based on the config.json
            # gpu_memory_utilization=0.9,
        )
    else:
        model = LLM(
            model=args.model_id,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": args.max_frame + 1, "video": 0},  # parts+frames
            dtype=torch.bfloat16,
            seed=args.seed,
            max_model_len=24000,  # kh: based on trial and error
            # gpu_memory_utilization=0.9,
        )

    return model, sampling_params


def format_input(args, examples, template_components, toy2instruction):
    logging.info("Prepare input ... ")

    inputs = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        prompt, images = "", []

        # prefix
        prompt += template_components["prefix"] + "\n"

        # parts
        prompt += template_components["parts"]
        prompt += "Image1: <image>\n\n"
        images.append(toy2instruction[example['toy_id']]['parts'])

        # assembly manual
        prompt += (
            template_components["manual"]["dot"].replace(
                "{dot}", toy2instruction[example["toy_id"]]["dot"]
            )
            + "\n"
        )

        # recording
        prompt += template_components["recording"]
        sampled_frames = sample_frame(
            example["video"]["start"],
            example["video"]["end"],
            args.max_frame,
        )
        images += sampled_frames
        prompt += (
            "".join([f"Frame{i+1}: <image>\n" for i in range(len(sampled_frames))]) + "\n"
        )

        # task
        prompt += template_components["task"].replace("{question}", example["question"])
        inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }
        )

    return inputs


def predict(model, sampling_params, inputs):
    logging.info("Prediction starts ... ")
    size = 40
    outputs = []
    for i in range(0, len(inputs), size):
        # open
        inputs_opened = []
        for _input in inputs[i : i + size]:
            _input_opened = {
                "prompt": _input["prompt"],
                "multi_modal_data": {
                    "image": [
                        Image.open(image) for image in _input["multi_modal_data"]["image"]
                    ]
                },
            }
            inputs_opened.append(_input_opened)

        # run inference
        _outputs = model.generate(inputs_opened, sampling_params)
        outputs += [x.outputs[0].text for x in _outputs]

    return outputs


def save(args, examples, inputs, predictions):
    new_examples = []
    for example, _input, prediction in zip(examples, inputs, predictions):
        new_example = deepcopy(example)
        new_example["prediction"] = {
            "text_prompt": _input["prompt"],
            "response": prediction,
        }
        new_examples.append(new_example)

    filepath_output = (
        args.dirpath_output
        / f"open_multimodal_{Path(args.model_id).name}_{args.resolution}_{args.max_frame}_{args.color}_{args.angle}.json"
    )

    output = {
        "metadata": {
            "data-created": get_date(),
            "model_id": args.model_id,
            "max_frame": args.max_frame,
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
    idx = 1
    logging.info(f"{idx=}")
    logging.info(inputs[idx]["prompt"])

    model, sampling_params = load_model(args)

    predictions = predict(model, sampling_params, inputs)

    save(args, examples, inputs, predictions)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code for Intern VL")
    
    parser.add_argument("--model_id", type=str, help="model id vision")
    
    # parser.add_argument("--filepath_input", type=Path, help="filepath for input")
    # parser.add_argument(
    #     "--filepath_instruction", type=Path, help="filepath for instruction"
    # )
    # parser.add_argument(
    #     "--dirpath_instruction_image", type=Path, help="dirpath for instruction (image)"
    # )
    # parser.add_argument(
    #     "--dirpath_parts_image", type=Path, help="dirpath for parts (image)"
    # )
    # parser.add_argument("--dirpath_frame", type=Path, help="dirpath for frames")
    parser.add_argument(
        "--filepath_template", type=Path, help="filepath for prompt template",
        default='.src/benchmark/template.yaml'
    )
    parser.add_argument(
        "--dirpath_output", type=Path, help="filepath for output",
        default='./output/prediction'
    )
    
    parser.add_argument("--temperature", type=float, help="temperature", default=0.0)
    parser.add_argument("--top_p", type=float, help="top p", default=1)
    parser.add_argument(
        "--max_tokens", type=int, help="max tokens to generate", default=256
    )
    parser.add_argument("--max_frame", type=int, help="max #frame", default=16)
    parser.add_argument("--angle", type=str, help="angle", default="C10115_rgb")
    parser.add_argument("--resolution", type=str, help="resolution", default="360p")
    parser.add_argument("--color", type=str, help="color", default="rgb")
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
            logging.FileHandler(
                args.dirpath_log / f"open_multimodal_internvl_{get_date()}.log"
            ),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
