"""
Evaluate (LLM-as-a-judge)

"""

from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
import logging
from pathlib import Path
import json
from tqdm import tqdm
import yaml

from src.utils.api import (
    call_openai_api_single,
    estimate_cost,
)

from src.utils.misc import (
    get_date,
)


def format_prev_steps(example, annotation):
    end = float(example["video"]["end"])

    output = ""
    for step in annotation["mistake"]["steps"]:
        if float(step["end"]) <= end:
            output += f"* {step['action']}\n"
        else:
            pass

    return output.strip()


def format_answers(example):
    output = ""
    for answer in example["answers"]:
        output += f"- {answer}\n"

    return output.strip()


def format_content(args, example, template_components, annotation):
    content = []

    content.append({"type": "text", "text": template_components["prefix"]})

    step = template_components["step"]
    step = step.replace("{previous_steps}", format_prev_steps(example, annotation))
    content.append({"type": "text", "text": step})

    content.append({"type": "text", "text": template_components["option"]})

    content.append({"type": "text", "text": template_components["note"]})

    # todo: do postprocess if needed
    predicted_answer = example["prediction"]["response"]
    task = template_components["task"]
    task = (
        task.replace("{question}", example["question"])
        .replace("{gold_answer}", format_answers(example))
        .replace("{predicted_answer}", predicted_answer)
    )
    content.append({"type": "text", "text": task})

    return content


def parse_feedback(feedback: str) -> tuple[str, str]:
    """
    parse feedback

    """

    splits = feedback.split("[Judge]")
    rationale, judge = splits

    return judge.strip(), rationale.strip()


def main(args):
    
    with open(args.file, "r") as f:
        examples = json.load(f)["examples"]

    with open(args.filepath_annotation, "r") as f:
        annotations = json.load(f)

    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    logging.info(f"#target examples: {len(examples)} ({args.template_type=})")

    logging.info("Call API")
    filepath_output = (
        args.dirpath_output
        / f"{Path(args.model_id).name}_{args.template_type}_{args.file.name}"
    )
    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    new_examples = []
    count_tokens = defaultdict(int)
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        annotation = annotations["examples"][example["sequence_id"]]

        content = format_content(args, example, template_components, annotation)

        if idx == 0:  # sanity check
            logging.info("content")
            for x in content:
                logging.info(x["text"])
        # sys.exit('stop')

        messages = [{"role": "user", "content": content}]
        response, _tokens = call_openai_api_single(
            model_id=args.model_id,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        new_example = deepcopy(example)
        new_example["evaluation"] = {
            "prompt": "\n".join([x["text"] for x in content]),
            "model_id": args.model_id,
            "template_type": args.template_type,
            "response": response,
        }
        try:
            judge, rationale = parse_feedback(response)
            new_example["evaluation"]["judge"] = judge
            new_example["evaluation"]["rationale"] = rationale
        except Exception as e:
            logging.warning(f"Error happened during postprocess: {e}")
        new_examples.append(new_example)

        count_tokens["input"] += _tokens["input"]
        count_tokens["output"] += _tokens["output"]

        with open(filepath_output, "w") as f:
            json.dump(new_examples, f, indent=4)
            f.write("\n")

    assert len(examples) == len(new_examples)

    cost = estimate_cost(args.model_id, count_tokens)
    logging.info(f"Estimated cost: ${cost:.4f}.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate")
    parser.add_argument("--file", type=Path, help="filepath to input data")

    parser.add_argument(
        "--filepath_annotation", type=Path, help="filepath to original example",
        default='./data/all_in_one_updated.json'
    )
    parser.add_argument(
        "--filepath_template", type=Path, help="filepath to template",
        default='./src/evaluation/templates_eval.yaml'
    )
    parser.add_argument(
        "--dirpath_output", type=Path, help="dirpath to output",
        default='./output/prediction'
    )
    # parser.add_argument(
    #     "--filepath_instruction", type=Path, help="filepath for instruction"
    # )
    
    
    parser.add_argument(
        "--template_type", type=str, help="template_type",
        default='ternary-step'
    )
    parser.add_argument(
        "--model_id", type=str, help="model id",
        default='gpt-4o-2024-11-20'
    )
    parser.add_argument("--temperature", type=float, help="temperature", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="max tokens to generate", default=512
    )
    parser.add_argument("--wait_time", type=int, help="API call wait time", default=0.5)
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / f"evaluate_{get_date()}.log"),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
