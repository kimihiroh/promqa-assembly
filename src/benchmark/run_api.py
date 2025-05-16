"""
Inference code for proprietary models via API
* Input: video (frames), instruction, question
* Output: answer

"""

from argparse import ArgumentParser
from copy import deepcopy
from collections import defaultdict
import json
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

from utils import (
    format_instruction,
    SUBSETS,
)

from utils_api import (
    format_content_openai,
    format_content_anthropic,
    format_content_google,
)

from src.utils.api import (
    call_openai_api_single,
    call_anthropic_api_single,
    call_google_api_single,
    estimate_cost,
)
from src.utils.misc import (
    get_date,
)


def format_input_and_call_api(args, examples, template_components, toy2instruction):
    logging.info("Create input ...")

    metadata = {
        "data-created": get_date(),
        "model_id": args.model_id,
        "max_frames": args.max_frames,
        "angle": args.angle,
    }

    filepath_output = (
        args.dirpath_output
        / f"{Path(args.model_id).name}_{args.resolution}_{args.max_frames}_{args.color}_{args.angle}.json"
    )
    if filepath_output.exists():
        with open(filepath_output, "r") as f:
            examples_prev = json.load(f)
        logging.info(
            f"Prev attempt exists. Restart from {len(examples_prev['examples'])}"
        )
    else:
        logging.info("Initial attempt")
        examples_prev = []

    new_examples, count_tokens = [], defaultdict(int)
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        if examples_prev and idx < len(examples_prev["examples"]):
            new_examples.append(examples_prev["examples"][idx])
            continue

        match args.model_id:
            case "gpt-4o-2024-11-20":
                content, text_prompt = format_content_openai(
                    args, example, template_components, toy2instruction
                )

                if idx == 0:
                    logging.info(f"Sanity check: prompt (text part only) for {idx=}")
                    logging.info(text_prompt)

                messages = [{"role": "user", "content": content}]
                response, count = call_openai_api_single(
                    model_id=args.model_id,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                count_tokens["input"] = count["input"]
                count_tokens["output"] = count["output"]
            case "claude-3-7-sonnet-20250219":
                content, text_prompt = format_content_anthropic(
                    args, example, template_components, toy2instruction
                )

                if idx == 0:
                    logging.info(f"Sanity check: prompt (text part only) for {idx=}")
                    logging.info(text_prompt)

                messages = [{"role": "user", "content": content}]
                response, count = call_anthropic_api_single(
                    model_id=args.model_id,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                count_tokens["input"] = count["input"]
                count_tokens["output"] = count["output"]
            case (
                "gemini-2.5-pro-exp-03-25"
                | "gemini-2.5-pro-preview-05-06"
                | "gemini-2.0-flash-001"
            ):
                content, text_prompt, uploaded_files = format_content_google(
                    args, example, template_components, toy2instruction
                )

                if idx == 0:
                    logging.info(f"Sanity check: prompt (text part only) for {idx=}")
                    logging.info(text_prompt)

                response, count = call_google_api_single(
                    model_id=args.model_id,
                    content=content,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    files=uploaded_files,
                )
                count_tokens["input"] = count["input"]
                count_tokens["output"] = count["output"]

            case _:
                logging.error(f"Undefined: {args.model_id=}")

        new_example = deepcopy(example)
        new_example["prediction"] = {
            "text_prompt": text_prompt,
            "response": response,
        }
        new_examples.append(new_example)

        output = {"metadata": metadata, "examples": new_examples}
        with open(filepath_output, "w") as f:
            json.dump(output, f, indent=4)
            f.write("\n")

    cost = estimate_cost(args.model_id, count_tokens)
    if examples_prev and "cost" in examples_prev["metadata"]:
        cost += examples_prev["metadata"]["cost"]
    metadata["cost"] = cost
    output = {"metadata": metadata, "examples": new_examples}
    with open(filepath_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")

    return None


def main(args):
    
    examples = []
    for toy_id in SUBSETS:
        examples += load_dataset("kimihiroh/promqa-assembly", toy_id, split='test')

    toy2instruction = format_instruction()

    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    logging.info("Prediction starts ...")
    format_input_and_call_api(args, examples, template_components, toy2instruction)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code")
    
    parser.add_argument("--model_id", type=str, help="model id")
    
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
    parser.add_argument(
        "--max_tokens", type=int, help="max tokens to generate", default=128
    )
    parser.add_argument("--max_frames", type=int, help="max frames to feed", default=10)
    parser.add_argument("--angle", type=str, help="angle", default="C10115_rgb")
    parser.add_argument("--resolution", type=str, help="resolution", default="360p")
    parser.add_argument("--color", type=str, help="color", default="rgb")
    parser.add_argument("--wait_time", type=int, help="API call wait time", default=10)
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
                args.dirpath_log
                / f"inference_{Path(args.model_id).name}_{get_date()}.log"
            ),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
