"""
Generate QAs for Assembly101

"""

from argparse import ArgumentParser
from copy import deepcopy
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random
import re
import yaml

from src.utils.api import (
    call_openai_api_single,
    call_anthropic_api_single,
    call_google_api_single,
    encode_image,
    estimate_cost,
)

TEMPLATE_TYPES = [
    ("next", True),
    ("next", False),
    ("missing", True),
    ("missing", False),
    ("order", True),
    ("order", False),
    ("past", None),
    ("misadjustment", None),
    ("general", None),
    ("current_general", None),
    ("location", True),
    ("location", False),
]


def get_prompt_template(template_type, components, granularity):
    type2prompt = defaultdict(lambda: defaultdict(None))

    for question_type, flag_info in TEMPLATE_TYPES:
        prompt = ""

        # prefix
        match template_type:
            case (
                "no-demonstration"
                | "text+image"
                | "question"
                | "inline-demonstration"
                | "default"
                | "question+image"
            ):
                prompt += f"{components['prefix']['default']}\n"
            case "answer" | "answer+image":
                prompt += f"{components['prefix']['answer']}\n"
            case _:
                logging.error(f"Undefined {template_type=}")

        # image
        match template_type:
            case (
                "no-demonstration"
                | "question"
                | "inline-demonstration"
                | "answer"
                | "default"
            ):
                pass
            case "text+image" | "question+image" | "answer+image":
                prompt += f"{components['image']}\n"
            case _:
                logging.error(f"Undefined {template_type=}")

        # performed steps
        match template_type:
            case (
                "no-demonstration"
                | "text+image"
                | "question"
                | "inline-demonstration"
                | "default"
                | "question+image"
            ):
                prompt += f"{components['performed_steps'][granularity]}\n"
            case "answer" | "answer+image":
                prompt += f"{components['performed_steps']['answer']}\n"
            case _:
                logging.error(f"Undefined {template_type=}")

        # graph error info
        match template_type:
            case "no-demonstration" | "text+image" | "inline-demonstration" | "default":
                instruction = components["graph_error_info"]["default"][question_type]
                if flag_info in [True, False]:
                    instruction = instruction[flag_info]
                else:
                    pass
            case "question" | "question+image":
                instruction = components["graph_error_info"]["question"][question_type]
            case "answer" | "answer+image":
                instruction = components["graph_error_info"]["answer"][question_type]
                if flag_info in [True, False]:
                    instruction = instruction[flag_info]
                else:
                    pass
            case _:
                logging.error(f"Undefined {template_type=}")

        if instruction:
            prompt += f"{instruction.strip()}\n\n"

        # task prompt
        match template_type:
            case (
                "no-demonstration"
                | "text+image"
                | "question"
                | "inline-demonstration"
                | "default"
                | "question+image"
            ):
                prompt += f"{components['task_prompt']['default'][question_type]}\n"
            case "answer" | "answer+image":
                prompt += f"{components['task_prompt']['answer']}\n"
            case _:
                logging.error(f"Undefined {template_type=}")

        # format
        match template_type:
            case "no-demonstration" | "text+image" | "inline-demonstration" | "default":
                prompt += f"{components['format']['default']}\n"
            case "question" | "question+image":
                prompt += f"{components['format']['question']}\n"
            case "answer" | "answer+image":
                prompt += f"{components['format']['answer']}\n"
            case _:
                logging.error(f"Undefined {template_type=}")

        # note
        match template_type:
            case "no-demonstration":
                prompt += f"{components['note']['default']}\n"
            case "question" | "question+image":
                prompt += f"{components['note']['question']}\n"
                if question_type in [
                    "next",
                    "missing",
                    "order",
                    "past",
                    "misadjustment",
                    "location",
                ]:
                    prompt += f"{components['example']['question']}\n"
                else:
                    pass
            case "answer" | "answer+image":
                prompt += f"{components['note']['answer']}\n"
            case "inline-demonstration":
                if question_type in [
                    "next",
                    "missing",
                    "order",
                    "past",
                    "misadjustment",
                    "location",
                ]:
                    prompt += f"{components['note']['inline-demonstration']}\n"
                else:
                    prompt += f"{components['note']['default']}\n"
            case "default" | "text+image":
                prompt += f"{components['note']['default']}\n"
                if question_type in [
                    "next",
                    "missing",
                    "order",
                    "past",
                    "misadjustment",
                    "location",
                ]:
                    prompt += f"{components['example']['default']}\n"
                else:
                    pass
            case _:
                logging.error(f"Undefined {template_type=}")

        # suffix
        prompt += f"{components['suffix']}"

        type2prompt[question_type][flag_info] = prompt

    return type2prompt


def get_question_type(example):
    question_type = example["type"]
    match question_type:
        case "next":
            flag = len(example["next_steps"]) > 0
        case "missing":
            flag = len(example["missing_steps"]) > 0
        case "order":
            flag = example["wrong_order_annotation"]
        case "past" | "misadjustment" | "general" | "current_general":
            flag = None
        case "location":
            flag = example["location_error"]
        case "confusion":
            question_type = "misadjustment"
            flag = None
        case _:
            logging.error(f"Undefined question type: {question_type}")

    return question_type, flag


def format_screw_info(is_screwed):
    output = ""
    if is_screwed is True:
        output = " (Estimated screw status: Completed)"
    elif is_screwed is False:
        output = " (Estimated screw status: Missed)"
    elif is_screwed is None:
        output = " (Screw not required)"
    else:
        logging.error(f"Undefined {is_screwed=}")

    return output


def format_fine_steps(current_step_id, mapping, fine_steps):
    output = ""
    if str(current_step_id) in mapping:
        for fine_step_id in mapping[str(current_step_id)]:
            output += f"\n    * {fine_steps[fine_step_id]['action']}"
    else:
        pass

    return output


def format_fine_steps_current(current_step_id, mapping, fine_steps):
    """
    trim within the current step

    later:
    * current: trimming_point is set as the index of fine_steps in each coarse step
    * new: set trimming point as index of all fine steps

    """
    output, trimming_point = "", None
    if str(current_step_id) in mapping:
        fine_step_ids = mapping[str(current_step_id)]
        if fine_step_ids:
            trimming_point = random.randint(1, len(fine_step_ids))
            for fine_step_id in fine_step_ids[:trimming_point]:
                output += f"\n    * {fine_steps[fine_step_id]['action']}"
    else:
        pass

    return output, trimming_point


def format_previous_steps(example, fine_steps, template_type, granularity, w_screw):
    output, trimming_point = "", None
    # add prev steps
    for step in example["previous_steps"]:
        if step["action"] == "start":
            continue

        output += f"* {step['action']}"

        if template_type not in ["question", "question+image"]:
            output += (
                f" (mistake: {step['mistake']})" if step["type"] == "mistake" else ""
            )
            output += format_screw_info(step["is_screwed"]) if w_screw else ""
        match granularity:
            case "coarse" | "coarse+fine_part":
                pass
            case "coarse+fine_all":
                # add info of screw needed or not
                if step["is_screwed"] in [True, False]:
                    output += " (screw-required step)"
                elif step["is_screwed"] is None:
                    output += " (no-screw step)"
                else:
                    logging.error(f"Undefined {step['is_screwed']=}")
                output += format_fine_steps(
                    step["step_id"],
                    example["mapping_coarse_to_fine"],
                    fine_steps,
                )
            case _:
                logging.error(f"Undefined {granularity=}")
        output += "\n"

    # add current step
    current_step = example["current_step"]
    output += f"* {current_step['action']}"
    if template_type not in ["question", "question+image"]:
        output += (
            f" (mistake: {current_step['mistake']})"
            if current_step["type"] == "mistake"
            else ""
        )
        output += format_screw_info(current_step["is_screwed"]) if w_screw else ""
    match granularity:
        case "coarse":
            pass
        case "coarse+fine_all" | "coarse+fine_part":
            if current_step["w_screw"]:
                output += " (screw-required step)"
            else:
                output += " (no-screw step)"
            _output, _trimming_point = format_fine_steps_current(
                current_step["step_id"],
                example["mapping_coarse_to_fine"],
                fine_steps,
            )
            output += _output
            trimming_point = _trimming_point
        case _:
            logging.error(f"Undefined {granularity=}")

    return output, trimming_point


def get_text_content(
    example,
    prompt_template,
    fine_steps,
    type2demonstrations,
    template_type,
    granularity,
    w_screw,
):
    previous_steps, trimming_point = format_previous_steps(
        example, fine_steps, template_type, granularity, w_screw
    )
    prompt = ""
    match example["type"]:
        case "next":
            next_steps = example["next_steps"]
            random.shuffle(next_steps)
            next_steps = "\n".join([f"* {x}" for x in next_steps])
            prompt = prompt_template.replace(
                "{previous_steps}", previous_steps
            ).replace("{next_steps}", next_steps.strip())
        case "missing":
            missing_steps = "\n".join([f"* {x}" for x in example["missing_steps"]])
            prompt = prompt_template.replace(
                "{previous_steps}", previous_steps
            ).replace("{missing_steps}", missing_steps.strip())
        case (
            "order"
            | "past"
            | "misadjustment"
            | "general"
            | "current_general"
            | "location"
        ):
            prompt = prompt_template.replace("{previous_steps}", previous_steps)
        case _:
            logging.error(
                f"Undefined question type: {example['type']} {prompt_template}"
            )

    if template_type in ["question", "question+image"]:
        if example["type"] in [
            "next",
            "missing",
            "order",
            "past",
            "misadustment",
            "general",
        ]:
            remaining_parts = "\n".join([f"* {x}" for x in example["remaining_parts"]])
            prompt = prompt.replace("{parts}", remaining_parts)
    if template_type in ["answer", "answer+image"]:
        prompt = prompt.replace("{question}", example["question"])

    if template_type == "inline-demonstration":
        if example["type"] in [
            "next",
            "missing",
            "order",
            "past",
            "misadjustment",
            "location",
        ]:
            demonstrations = type2demonstrations[example["type"]]
            demonstration = random.choice(demonstrations)
            prompt = prompt.replace("{demonstration}", f'"{demonstration}"')
    if template_type in ["default", "text+image", "question", "question+image"]:
        if example["type"] in [
            "next",
            "missing",
            "order",
            "past",
            "misadjustment",
            "location",
        ]:
            demonstrations = type2demonstrations[example["type"]]
            demonstrations_sampled = random.sample(demonstrations, 2)
            prompt = prompt.replace(
                "{demonstration1}", demonstrations_sampled[0]
            ).replace("{demonstration2}", demonstrations_sampled[1])

    content = {"type": "text", "text": prompt}

    return content, prompt, trimming_point


def get_image_content(filepath):
    output = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(filepath)}",
        },
    }
    return output


def postprocess(template_type, generation):
    if not generation:
        logging.warning(f"Empty {generation=}")
        return []

    output = []
    match template_type:
        case (
            "no-demonstration"
            | "question"
            | "inline-demonstration"
            | "text+image"
            | "default"
            | "question+image"
        ):
            qas = []
            matches = re.findall(r"\* (.*?)\n((?: {2,5}\* .*?\n)*)", generation + "\n")
            for question, answers in matches:
                qa = {"question": question, "answers": []}
                for raw_answer in answers.splitlines():
                    _matches = re.findall(r" {2,5}\* (.*)", raw_answer)
                    if len(_matches) != 1:
                        logging.warning(f"Incorrect parsing for answer: {raw_answer}")
                    else:
                        qa["answers"].append(_matches[0])
                qas.append(qa)

            if len(qas) == 0:
                qas.append({"question": None, "answers": [None]})

            if len(qas) != 3:
                logging.warning(
                    f"{len(qas)} (!=3) QAs were extracted from:\n"
                    f"generation={generation} qas={qas}\n"
                )
            output = qas
        case "answer" | "answer+image":
            answers = []
            matches = re.findall(r"\*\s{1,5}(.*?)(?:\n|$)", generation)
            for answer in matches:
                if answer.strip():
                    answers.append(answer)

            if not answers:
                logging.warning(f"No answers were extracted from {generation}")
            output = answers
        case _:
            logging.error(f"Undefined {template_type=} for postprocess")

    return output


def main(args):
    random.seed(args.seed)

    # load sampled examples
    with open(args.filepath_input, "r") as f:
        examples = json.load(f)

    # load annotations
    with open(args.filepath_all, "r") as f:
        annotations = json.load(f)

    # load prompt tempaltes
    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    type2prompt = get_prompt_template(
        args.template_type, template_components, args.granularity
    )

    type2demonstrations = template_components["demonstrations"]

    messages_list, prompts = [], []
    for example in examples["examples"]:
        contents = []

        question_type, question_type_flag = get_question_type(example)

        text_content, text_prompt, trimming_point = get_text_content(
            example,
            type2prompt[question_type][question_type_flag],
            annotations["examples"][example["metadata"]["sequence_id"]]["fine"][
                "steps"
            ],
            type2demonstrations,
            args.template_type,
            args.granularity,
            args.w_screw,
        )
        contents.append(text_content)
        prompts.append([text_prompt, trimming_point])

        match args.template_type:
            case (
                "no-demonstration"
                | "question"
                | "answer"
                | "inline-demonstration"
                | "default"
            ):
                pass
            case "text+image" | "question+image" | "answer+image":
                image_content = get_image_content(
                    args.dirpath_image / f"{example['metadata']['toy_id']}-all.png"
                )
                contents.append(image_content)
            case _:
                logging.error(f"Undefined {args.template_type=}")

        messages_list.append([{"role": "user", "content": contents}])

    # manual check
    for idx in [2]:
        logging.info(f"[sanity check] prompts[{idx}]")
        logging.info(prompts[idx][0])
    logging.info(f"#requests: {len(messages_list)}/{len(examples['examples'])}")

    logging.info("API call starts")
    responses = []
    total_tokens = defaultdict(int)
    for messages in messages_list:
        match args.model_id:
            case "gpt-4o-2024-11-20" | "o1-2024-12-17" | "o3-mini-2025-01-31":
                response, output_tokens = call_openai_api_single(
                    model_id=args.model_id,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                responses.append(response)
                for key, count in output_tokens.items():
                    total_tokens[key] += count
            case "claude-3-7-sonnet-20250219":
                response, output_tokens = call_anthropic_api_single(
                    model_id=args.model_id,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                responses.append(response)
                for key, count in output_tokens.items():
                    total_tokens[key] += count
            case "gemini-2.5-pro-exp-03-25" | "gemini-2.0-flash-001":
                response, output_tokens = call_google_api_single(
                    model_id=args.model_id,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    files=[],
                )
                responses.append(response)
                for key, count in output_tokens.items():
                    total_tokens[key] += count
            case _:
                logging.error(f"Undefined {args.model_id=}")

    estimate_cost(args.model_id, total_tokens)

    logging.info("Save outputs")
    filepath_output = (
        args.dirpath_output / f"{args.filepath_input.stem}.{args.template_type}"
        f".{args.model_id}.{args.granularity}.{args.w_screw}.json"
    )
    metadata = {
        "data-created": datetime.today().strftime("%Y-%m-%d"),
        "model_id": args.model_id,
        "filepath_input": str(args.filepath_input),
        "template_type": args.template_type,
        "granularity": args.granularity,
        "w_screw": args.w_screw,
    }
    # saving raw output
    output = {"metadata": metadata, "prompts": prompts, "responses": responses}
    with open(filepath_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")

    # combine input&output
    # assert len(prompts) == len(responses) == len(examples["examples"])
    new_examples = []
    for example, prompt, response in zip(examples["examples"], prompts, responses):
        # parse
        parsed_output = postprocess(args.template_type, response)
        # create new example
        new_example = deepcopy(example)

        match args.template_type:
            case (
                "no-demonstration"
                | "question"
                | "inline-demonstration"
                | "text+image"
                | "default"
                | "question+image"
            ):
                new_example["generation"] = {
                    "prompt": prompt,
                    "response": response,
                    "model_id": args.model_id,
                    "template_type": args.template_type,
                    "granularity": args.granularity,
                    "w_screw": args.w_screw,
                    "qas": parsed_output,
                }
                # sample one and store
                qa = random.choice(parsed_output)
                new_example["question"] = qa["question"]
                new_example["answers"] = qa["answers"]
            case "answer" | "answer+image":
                if new_example["answers"]:
                    logging.warning("this should be empty")
                new_example["generation"]["prompt-answer"] = prompt
                new_example["generation"]["response-answer"] = response
                new_example["answers"] = parsed_output
            case _:
                logging.error(f"Undefined {args.template_type=} for saving")

        new_examples.append(new_example)

    # save combined version
    output = {"metadata": metadata, "examples": new_examples}
    with open(filepath_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate QAs for Assembly101")
    parser.add_argument("--filepath_input", type=Path, help="filepath to input data")
    parser.add_argument("--filepath_all", type=Path, help="filepath to all annotation")
    parser.add_argument("--dirpath_image", type=Path, help="dirpath to toy image")
    parser.add_argument(
        "--filepath_template", type=Path, help="filepath to prompt template"
    )
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument("--model_id", type=str, help="model id")
    parser.add_argument("--template_type", type=str, help="prompt template type")
    parser.add_argument(
        "--w_screw", action="store_true", help="True if screw info in prompt"
    )
    parser.add_argument("--granularity", type=str, help="context granularity")
    parser.add_argument("--temperature", type=float, help="temperature", default=0.5)
    parser.add_argument("--max_tokens", type=int, help="max #tokens", default=512)
    parser.add_argument("--max_frames", type=int, help="max frames to feed", default=50)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                args.dirpath_log / "generate_qas"
                f"{args.filepath_input.stem}.{args.model_id}."
                f"{args.granularity}.{args.w_screw}.log"
            ),
        ],
    )

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    logging.info(f"Arguments: {vars(args)}")

    main(args)
