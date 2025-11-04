from argparse import ArgumentParser
import json
import logging
from pathlib import Path

from src.utils.misc import get_date, format_time


def identify_end(example, annotation):
    """check if the end is based on coarse of fine"""

    end = 0
    if example["generation"]["prompt"][1]:  # based on fine
        id_fine_in_coarse = example["generation"]["prompt"][1]
        id_fine = example["mapping_coarse_to_fine"][
            str(example["current_step"]["step_id"])
        ][id_fine_in_coarse - 1]
        step_fine = annotation["fine"]["steps"][id_fine]
        end = step_fine["end"]
    else:  # based on coarse
        end = example["current_step"]["end"]

    return end


def format_caption(steps):
    caption = "WEBVTT\n\n"
    for step in steps:
        caption += f"{format_time(step['start'])} --> {format_time(step['end'])}\n"
        caption += f"{step['action']}"
        if step["label"] and step["label"] != "correct":
            caption += f" ({step['label']})"
        caption += "\n\n"

    return caption


def format_actions(example, annotation):
    """
    action list format
    * [time ~ range] coarsre
        * [time ~ range] fine
        * ...
    * ...

    """

    output = ""
    for step in example["previous_steps"] + [example["current_step"]]:
        if step["action"] == "start":
            continue

        output += f"* [{format_time(step['start'], 'short')} -> {format_time(step['end'], 'short')}]"
        output += f" {step['action']}"
        if step["type"] != "correct":
            output += f" ({step['mistake']})"
        output += "\n"
    return output


def main(args):
    with open(args.filepath_input, "r") as f:
        data = json.load(f)

    with open(args.filepath_all, "r") as f:
        annotations = json.load(f)

    new_examples = []
    logging.info("Formatting ...")
    for example in data["examples"]:
        annotation = annotations["examples"][example["metadata"]["sequence_id"]]
        filename_graph_w_status = (
            f"{example['metadata']['user_id']}"
            f"-{example['metadata']['toy_id']}"
            f"-{example['metadata']['recording_id']}"
            f"-{example['current_step']['step_id']}.png"
        )

        new_examples.append(
            {
                "sequence_id": example["metadata"]["sequence_id"],
                "video": {
                    "angles": annotation["recording"]["angles"],
                    "start": example["metadata"]["start"],
                    "end": identify_end(example, annotation),
                    "caption": format_caption(annotation["mistake"]["steps"]),
                },
                "filepath_part": f"{example['metadata']['toy_id']}-all.png",
                "actions": format_actions(example, annotation),
                "filepath_graph": filename_graph_w_status,
                "filepath_graph_original": f"{example['metadata']['toy_id']}.png",
                "question": example["question"],
                "answers": example["answers"],
                "verification": {
                    "question": {
                        "valid": None,
                        "multimodal": None,
                        "procedural": None,
                    },
                    "answers": [{"correct": None} for x in example["answers"]],
                    "comment": None,
                },
            }
        )
    logging.info("Done.")

    output = {
        "metadata": {
            "data-created": get_date(granularity="day"),
            "total": len(new_examples),
        },
        "examples": new_examples,
    }
    filename_output = f"{args.filepath_input.stem.split('.')[0]}.json"
    with open(args.dirpath_output / filename_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Postprocess generated QAs")
    parser.add_argument("--filepath_input", type=Path, help="filepath to input data")
    parser.add_argument("--filepath_all", type=Path, help="filepath to all annotation")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "postprocess_qa.log"),
        ],
    )

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    logging.info(f"Arguments: {vars(args)}")

    main(args)
