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
from google import genai
import os

from utils import (
    format_instruction,
)

from utils_api import (
    format_content_openai,
    format_content_anthropic,
    format_content_google,
    format_content_openai_cot,
    format_content_anthropic_cot,
    format_content_google_cot,
    format_content_openai_text,
    format_content_anthropic_text,
    format_content_google_text,
    format_content_openai_response_api,
)

from src.utils.api import (
    call_openai_api_single,
    call_anthropic_api_single,
    call_google_api_single,
    call_openai_api_single_reasoning,
    call_anthropic_api_single_reasoning,
    call_google_api_single_reasoning,
    estimate_cost,
)
from src.utils.misc import (
    get_date,
)


def format_input_and_call_api(args, examples, template_components, toy2instruction):
    logging.info("Create input ...")

    metadata = {
        "data-created": get_date(),
        "input": args.filepath_input.name,
        "model_id": args.model_id,
        "max_frames": args.max_frames,
        "angle": args.angle,
    }

    if args.mode == "reasoning" and args.model_id == "gpt-5-2025-08-07":
        filepath_output = (
            args.dirpath_output
            / f"{Path(args.model_id).name}_{args.mode}_{args.reasoning_effort}"
            f"_{args.resolution}_{args.max_frames}_{args.color}_{args.angle}"
            f"_{args.filepath_input.name}"
        )
    else:
        filepath_output = (
            args.dirpath_output
            / f"{Path(args.model_id).name}_{args.mode}_{args.resolution}"
            f"_{args.max_frames}_{args.color}_{args.angle}_{args.filepath_input.name}"
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

        response, thinking = "", ""
        match args.model_id:
            case "gpt-4o-2024-11-20":
                if args.mode == "cot":
                    content, text_prompt = format_content_openai_cot(
                        args, example, template_components, toy2instruction
                    )
                elif args.mode == "text":
                    content, text_prompt = format_content_openai_text(
                        args, example, template_components, toy2instruction
                    )
                else:
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
                count_tokens["input"] += count["input"]
                count_tokens["output"] += count["output"]
            case "gpt-5-2025-08-07":
                content, text_prompt = format_content_openai_response_api(
                    args, example, template_components, toy2instruction
                )
                if idx == 0:
                    logging.info(f"Sanity check: prompt (text part only) for {idx=}")
                    logging.info(text_prompt)

                messages = [{"role": "user", "content": content}]
                response, thinking, count = call_openai_api_single_reasoning(
                    model_id=args.model_id,
                    messages=messages,
                    reasoning_effort=args.reasoning_effort,
                )
                count_tokens["input"] += count["input"]
                count_tokens["output"] += count["output"]
            case "claude-3-7-sonnet-20250219":
                if args.mode == "cot":
                    content, text_prompt = format_content_anthropic_cot(
                        args, example, template_components, toy2instruction
                    )
                elif args.mode == "text":
                    content, text_prompt = format_content_anthropic_text(
                        args, example, template_components, toy2instruction
                    )
                else:
                    content, text_prompt = format_content_anthropic(
                        args, example, template_components, toy2instruction
                    )

                if idx == 0:
                    logging.info(f"Sanity check: prompt (text part only) for {idx=}")
                    logging.info(text_prompt)

                messages = [{"role": "user", "content": content}]
                if args.mode == "reasoning":
                    response, thinking, count = call_anthropic_api_single_reasoning(
                        model_id=args.model_id,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        budget_tokens=args.budget_tokens,
                    )
                else:
                    response, count = call_anthropic_api_single(
                        model_id=args.model_id,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                count_tokens["input"] += count["input"]
                count_tokens["output"] += count["output"]
            case (
                "gemini-2.5-pro-exp-03-25"
                | "gemini-2.5-pro-preview-05-06"
                | "gemini-2.0-flash-001"
                | "gemini-2.5-pro"
            ):
                if args.mode == "cot":
                    content, text_prompt, uploaded_files = format_content_google_cot(
                        args, example, template_components, toy2instruction
                    )
                elif args.mode == "text":
                    content, text_prompt, uploaded_files = format_content_google_text(
                        args, example, template_components, toy2instruction
                    )
                else:
                    content, text_prompt, uploaded_files = format_content_google(
                        args, example, template_components, toy2instruction
                    )

                if idx == 0:
                    logging.info(f"Sanity check: prompt (text part only) for {idx=}")
                    logging.info(text_prompt)

                if args.mode == "reasoning":
                    response, count = call_google_api_single_reasoning(
                        model_id=args.model_id,
                        content=content,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        files=uploaded_files,
                    )
                else:
                    response, count = call_google_api_single(
                        model_id=args.model_id,
                        content=content,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        files=uploaded_files,
                    )
                count_tokens["input"] += count["input"]
                count_tokens["output"] += count["output"]

                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                for uploaded_file in uploaded_files:
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except Exception as e:
                        logging.warning(f"Exception in file deletion: {e}")

            case _:
                logging.error(f"Undefined: {args.model_id=}")

        new_example = deepcopy(example)
        new_example["prediction"] = {
            "text_prompt": text_prompt,
            "response": response,
            "thinking": thinking,
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
    with open(args.filepath_input, "r") as f:
        examples = json.load(f)

    toy2instruction = format_instruction(
        args.filepath_instruction,
        args.dirpath_instruction_image,
        args.dirpath_parts_image,
    )

    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    logging.info("Prediction starts ...")
    format_input_and_call_api(args, examples, template_components, toy2instruction)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code")
    parser.add_argument("--filepath_input", type=Path, help="filepath for input")
    parser.add_argument(
        "--filepath_instruction", type=Path, help="filepath for instruction"
    )
    parser.add_argument(
        "--dirpath_instruction_image", type=Path, help="dirpath for instruction (image)"
    )
    parser.add_argument(
        "--dirpath_parts_image", type=Path, help="dirpath for parts (image)"
    )
    parser.add_argument("--dirpath_frame", type=Path, help="dirpath for frames")
    parser.add_argument(
        "--filepath_template", type=Path, help="filepath for prompt template"
    )
    parser.add_argument("--dirpath_output", type=Path, help="filepath for output")
    parser.add_argument("--model_id", type=str, help="model id")
    parser.add_argument("--temperature", type=float, help="temperature", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="max tokens to generate", default=4608
    )
    parser.add_argument(
        "--budget_tokens",
        type=int,
        help="max budget tokens for reasoning",
        default=4098,
    )
    parser.add_argument("--max_frames", type=int, help="max frames to feed", default=10)
    parser.add_argument("--angle", type=str, help="angle", default="C10118_rgb")
    parser.add_argument("--resolution", type=str, help="resolution", default="360p")
    parser.add_argument("--color", type=str, help="color", default="rgb")
    parser.add_argument("--wait_time", type=int, help="API call wait time", default=10)
    parser.add_argument(
        "--mode", type=str, help="default, cot, text, reasoning", default="default"
    )
    parser.add_argument(
        "--reasoning_effort", type=str, help="reasoning effort", default="minimal"
    )
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

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
