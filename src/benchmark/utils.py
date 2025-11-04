"""
helper functions

"""

import logging
import pydot
import re
import json


def format_instruction(
    filepath_text, dirpath_instruction_image=None, dirpath_parts_image=None
):
    logging.info("Load instructions ... ")

    toy2instruction = {}
    with open(filepath_text, "r") as f:
        data = json.load(f)

    for annotation in data["examples"]:
        toy_id = annotation["toy_id"]

        G = pydot.Dot(graph_type="digraph")
        action_id2description = {}
        for action in annotation["nodes"]:
            idx = str(action["id"])
            if "checked" in action["data"] and action["data"]["checked"]:
                description = f"{action['data']['label']} w/ screw"
            else:
                description = f"{action['data']['label']}"
            action_id2description[idx] = description
            node = pydot.Node(f"{description}")
            G.add_node(node)

        for edge in annotation["edges"]:
            edge = pydot.Edge(
                action_id2description[str(edge["source"])],
                action_id2description[str(edge["target"])],
            )
            G.add_edge(edge)

        toy2instruction[toy_id] = {
            "dot": G.to_string().strip(),
            "dag": dirpath_instruction_image / f"{toy_id}.png"
            if dirpath_instruction_image
            else None,
            "parts": dirpath_parts_image / annotation["filepath_image"]
            if dirpath_parts_image
            else None,
        }

    return toy2instruction


def extract_index(filepath):
    return int(re.search(r"\d+", filepath.stem).group())


def sample_frame(dirpath, start, end, max_frames):
    target_filepaths = []
    for filepath in dirpath.glob("*.png"):
        idx = int(filepath.stem)
        if float(start) <= idx <= float(end):
            target_filepaths.append(filepath)

    num_frames = len(target_filepaths)

    # e.g., 70 frames, max 25 => rate: 1 frame per every 3 frames
    if num_frames > max_frames:
        if num_frames % max_frames == 0:
            rate_inverse = num_frames // max_frames
        else:
            rate_inverse = (num_frames // max_frames) + 1
    else:
        rate_inverse = 1
    filepaths_frame_sorted = sorted(target_filepaths, key=extract_index)

    sampled_filepaths = []
    # note: "reversed" to make sure the last frame is included in the input
    for idx, filepath_frame in enumerate(reversed(filepaths_frame_sorted)):
        # change sample rate
        if idx % rate_inverse == 0:
            sampled_filepaths.insert(0, filepath_frame)

    assert len(sampled_filepaths) <= max_frames

    return sampled_filepaths


def format_steps(steps: list, w_error: bool = False) -> str:
    output = ""
    for step in steps:
        output += f"- {step['description']}\n"
        if w_error and "errors" in step:
            for error in step["errors"]:
                output += f"    - [{error['tag']}] {error['description']}\n"

    return output.strip()


def get_text_content_evaluation(
    model_id: str,
    template_type: str,
    components: dict,
    name2recipe: dict,
    example: dict,
) -> tuple[list, str]:
    """

    todo: merge with get_text_content above
    """
    prompt = None
    template = components["prefix"]
    if "binary" in template_type:
        template += f"\n{components['option']['binary']}"
    elif "ternary" in template_type:
        template += f"\n{components['option']['ternary']}"
    template += f"\n{components['note']['default']}"
    if "recipe" in template_type:
        template += f"\n{components['note']['recipe']}"
    if "step" in template_type:
        template += f"\n{components['note']['step']}"
    template += f"\n{components['task']}"

    steps = example["previous_steps"] + [example["current_step"]]

    question = f"- {example['question']}"
    gold_answers = "\n".join([f"- {answer}" for answer in example["answers"]]).strip()
    if "human_answer" in example:
        predicted_answer = f"- {example['human_answer']}"
    else:
        predicted_answer = f"- {example['prediction']['response']}"

    prompt = (
        template.replace("{activity_name}", example["activity_name"])
        .replace("{step_information}", format_steps(steps=steps, w_error=True))
        .replace("{recipe}", name2recipe[example["activity_name"]]["dot"])
        .replace("{question}", question)
        .replace("{gold_answer}", gold_answers)
        .replace("{predicted_answer}", predicted_answer)
    )

    if "gpt" in model_id:
        content = [
            {
                "type": "text",
                "text": prompt.strip(),
            }
        ]
    elif "claude" in model_id:
        content = [
            {
                "type": "text",
                "text": prompt.strip(),
            }
        ]
    elif "gemini" in model_id:
        content = [prompt]
    else:
        logging.error(f"Undefined {model_id=}")
        content = []

    return content, prompt
