"""
Create examples from Assembly101
* create examples from mistake annotations
* align annotations from coarse annotations
* load recording's meta information

"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime

from utils_create_example import (
    to_dict,
    create_graph,
    format_metadata,
    create_init_step,
    format_example_base,
    format_example_next,
    format_example_missing,
    format_example_order,
    format_example_past,
    format_example_misadjustment,
    format_example_general,
    format_example_location,
    format_example_current_general,
    check_overlap_coarse,
)

QUESTION_TYPES = [
    "next",
    "missing",
    "order",
    "past",
    "misadjustment",
    "general",
    "current_general",
    "location",
]


def stats(examples):
    logging.info(f"#exmaples: {len(examples)}")

    # count: noisy
    count = defaultdict(lambda: defaultdict(int))
    for example in examples:
        count["toy_id"][example["metadata"]["toy_id"]] += 1
        idx = (
            f"{example['metadata']['user_id']}-{example['metadata']['toy_id']}"
            f"-{example['metadata']['recording_id']}"
        )
        count["recording"][idx] += 1
        count["is_noisy"][example["is_noisy"]] += 1
        count["action_type"][example["current_step"]["type"]] += 1
        count["question_type"][example["type"]] += 1
        count["num_prev_steps"][len(example["previous_steps"])] += 1
        match example["type"]:
            case "next":
                count["next"][len(example["next_steps"])] += 1
            case "missing":
                count["missing"][len(example["missing_steps"])] += 1
            case "order":
                count["order"][example["wrong_order_annotation"]] += 1
            case "past":
                count["past"][bool(example["error_accumulation"])] += 1
            case "misadjustment":
                count["misadjustment"][example["misadjustment"]] += 1
            case "location":
                count["location"][example["location_error"]] += 1
        count["is_screwed"][example["current_step"]["is_screwed"]] += 1

    count = to_dict(count)
    logging.info(f"{len(count['toy_id'])=}")
    logging.info(f"{len(count['recording'])=}")
    logging.info(f"{count['is_noisy']=}")
    logging.info(f"{count['action_type']=}")
    logging.info(f"{count['question_type']=}")
    logging.info(f"{count['num_prev_steps']=}")
    logging.info(f"{count['is_screwed']=}")
    logging.info(f"{count['next']=}")
    logging.info(f"{count['missing']=}")
    logging.info(f"{count['order']=}")
    logging.info(f"{count['past']=}")
    logging.info(f"{count['misadjustment']=}")
    logging.info(f"{count['location']=}")

    return None


def main(args):
    # load graph annotation
    with open(args.filepath_graph, "r") as f:
        annotations_graph = json.load(f)

    # load example annotation
    with open(args.filepath_input, "r") as f:
        annotations = json.load(f)

    target_graphs = {}
    for graph in annotations_graph["examples"]:
        target_graphs[graph["toy_id"]] = create_graph(graph)

    target_annotations = []
    for sequence_id, annotation in annotations["examples"].items():
        target_annotations.append([sequence_id, annotation])

    """
    [target]
    * process-level
      - next: (extract from graph)
      - missing: (extract from graph)
      - order: wrong order
      - past: prev one is mistake
      - misadjustment: shouldn't have happened
      - general
    * step-specific
      - location/position: wrong position
      - general

    two sets
    * with screws
    * without screws
    """
    examples = []
    for sequence_id, annotation in target_annotations:
        start = annotation["mistake"]["steps"][0]["start"]
        metadata = format_metadata(sequence_id, annotation)
        graph = target_graphs[metadata["toy_id"]]

        previous_steps = [create_init_step()]

        for idx, step in enumerate(annotation["mistake"]["steps"]):
            example_base, previous_step = format_example_base(
                idx,
                sequence_id,
                annotation,
                previous_steps,
                graph,
            )

            flag_too_short = (step["end"] - start) < 10  # hard-coded
            flag_overlap = check_overlap_coarse(idx, annotation["mistake"]["steps"])

            if not flag_too_short and not flag_overlap:
                # process-level
                ## next
                examples.append(format_example_next(graph, example_base))
                ## missing
                examples.append(format_example_missing(graph, example_base))
                ## order
                examples.append(format_example_order(graph, example_base))
                ## past
                examples.append(format_example_past(graph, example_base))
                ## misadjustment
                examples.append(format_example_misadjustment(graph, example_base))
                ## general
                examples.append(format_example_general(graph, example_base))

                # step-specific
                ## location
                examples.append(format_example_location(graph, example_base))
                ## general
                examples.append(format_example_current_general(graph, example_base))

    stats(examples)

    output = {
        "metadata": {
            "data-created": datetime.today().strftime("%Y-%m-%d"),
            "total": len(examples),
        },
        "examples": examples,
    }
    with open(args.filepath_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create exmaples from Assembly101")
    parser.add_argument("--filepath_input", type=Path, help="filepath to input")
    parser.add_argument("--filepath_graph", type=Path, help="filepath to graph")
    parser.add_argument("--filepath_output", type=Path, help="filepath to output")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "create_example.log"),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
