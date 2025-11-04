"""
Format assembly101 annotation
* collect into one file

"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import yaml
from collections import defaultdict
from datetime import datetime

from src.utils.misc import to_dict, format_time

camera_set1 = {
    "C10095_rgb.mp4",
    "C10118_rgb.mp4",
    "C10379_rgb.mp4",
    "C10395_rgb.mp4",
    "C10115_rgb.mp4",
    "C10119_rgb.mp4",
    "C10390_rgb.mp4",
    "C10404_rgb.mp4",
    "HMC_84346135_mono10bit.mp4",
    "HMC_84355350_mono10bit.mp4",
    "HMC_84347414_mono10bit.mp4",
    "HMC_84358933_mono10bit.mp4",
}
camera_set2 = {
    "C10095_rgb.mp4",
    "C10118_rgb.mp4",
    "C10379_rgb.mp4",
    "C10395_rgb.mp4",
    "C10115_rgb.mp4",
    "C10119_rgb.mp4",
    "C10390_rgb.mp4",
    "C10404_rgb.mp4",
    "HMC_21176875_mono10bit.mp4",
    "HMC_21176623_mono10bit.mp4",
    "HMC_21110305_mono10bit.mp4",
    "HMC_21179183_mono10bit.mp4",
}


def get_stats(id2example, keys):
    logging.info(f"#sequence: {len(id2example)}")

    if "recording" in keys:
        logging.info(
            f"#recording id: {len(set([x['recording_id'] for x in id2example.values()]))}"
        )
        logging.info(f"#user: {len(set([x['user_id'] for x in id2example.values()]))}")
        logging.info(f"#toy: {len(set([x['toy_id'] for x in id2example.values()]))}")
        count_camera_set = {"1": 0, "2": 0}
        for key, value in id2example.items():
            if len(value["angles"]) != 12:
                logging.debug(f"#angle in {key}: {len(value['angles'])}")
            else:
                if set(value["angles"]) == camera_set1:
                    count_camera_set["1"] += 1
                elif set(value["angles"]) == camera_set2:
                    count_camera_set["2"] += 1
                else:
                    logging.error("undefined camera set: this should be reached")
        logging.info(f"camera_set: {count_camera_set}")

    if "mistake" in keys:
        avg = sum([len(x["steps"]) for x in id2example.values()]) / len(id2example)
        logging.info(f"Avg. num steps: {avg:.1f}")

        count = defaultdict(lambda: defaultdict(int))
        for idx, example in id2example.items():
            count["with_error"][example["with_error"]] += 1
            for step in example["steps"]:
                count["verb"][step["verb"]] += 1
                count["error-label"][step["label"]] += 1
                count["remark"][step["remark"]] += 1
        logging.info(f"w/ error: {dict(count['with_error'])}")
        logging.info(f"verb: {dict(count['verb'])}")
        logging.info(f"label: {dict(count['error-label'])}")
        logging.info(f"remark: {dict(count['remark'])}")

        durations = [
            x["steps"][-1]["end"] - x["steps"][0]["start"] for x in id2example.values()
        ]
        logging.info(
            f"Avg. duration (assembly): {format_time(sum(durations)/len(durations))}"
        )

    if "coarse" in keys:
        nums_step = []
        durations = []
        for example in id2example.values():
            if "steps" in example:
                nums_step.append(len(example["steps"]))
                durations.append(
                    example["steps"][-1]["end"] - example["steps"][0]["start"]
                )
        avg = sum(nums_step) / len(nums_step)
        logging.info(f"Avg. num steps: {avg:.1f}")
        logging.info(
            f"Avg. duration (assembly): {format_time(sum(durations)/len(durations))}"
        )

    if "fine" in keys:
        avg = sum([len(x["steps"]) for x in id2example.values()]) / len(id2example)
        logging.info(f"Avg. num steps: {avg:.1f}")

        durations = [
            x["steps"][-1]["end"] - x["steps"][0]["start"] for x in id2example.values()
        ]
        logging.info(
            f"Avg. duration (all): {format_time(sum(durations)/len(durations))}"
        )

    if "all" in keys:
        logging.info(
            f"#toy: {len(set([x['recording']['toy_id'] for x in id2example.values()]))}"
        )

    return None


def get_sequences_from_recording(dirpath):
    sequences = {}
    for folderpath in dirpath.glob("nusar*"):
        assert "both" in folderpath.stem

        match = re.search(r"(\d{4})-([^_]+)_(\d{4})", folderpath.stem)
        user_id = match.group(1)
        toy_id = match.group(2)
        user_id2 = match.group(3)

        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{6})", folderpath.stem)
        recording_id = match.group(1)

        # use first user id is okay even when user id2 is different
        if user_id != user_id2:
            logging.debug(
                f"user id mismatch, {user_id} != {user_id2}, in {folderpath.stem}"
            )

        sequences[folderpath.stem] = {
            "user_id": user_id,
            "toy_id": toy_id,
            "recording_id": recording_id,
            "angles": sorted([x.name for x in folderpath.glob("*.mp4")]),
        }

    return to_dict(sequences)


mistake_description_correction_mapping = {
    "previous one was mistake": "previous one is a mistake",
    "shouln't have happened": "shouldn't have happened",
    "previous is a mistake": "previous one is a mistake",
    "previous one is mistake": "previous one is a mistake",
    "worng order": "wrong order",
}


def format_action_label(verb, this, that):
    action = ""
    if verb == "attach":
        action = f"{verb} {this} to {that}"
    elif verb == "detach":
        action = f"{verb} {this} from {that}"
    elif verb == "position":
        action = f"{verb} {this} in. {that}"
    else:
        logging.error(f"Undefined verb: {verb}")

    return action


def get_sequences_from_mistake(dirpath):
    sequences = {}
    for filepath in dirpath.glob("*.csv"):
        df = pd.read_csv(filepath, header=None)
        df = df.replace({np.nan: None})

        steps = []
        flag_with_error = False
        for idx, row in df.iterrows():
            verb = row[2]
            if row[2] == "interior":  # manual correction
                verb = "detach"
                logging.debug(
                    f"correct {row[2]} to {verb} on the step {idx} in {filepath.stem}"
                )

            step = {
                "start": int(row[0]) / 30,  # note: unit is now second
                "end": int(row[1]) / 30,
                "action": format_action_label(verb, row[3], row[4]),
                "verb": verb,
                "this": row[3],
                "that": row[4],
                "label": row[5],
            }
            if df.shape[1] == 7:
                if row[6] in mistake_description_correction_mapping:
                    step["remark"] = mistake_description_correction_mapping[row[6]]
                else:
                    step["remark"] = row[6]
            else:
                step["remark"] = None

            if step["remark"]:
                flag_with_error = True

            steps.append(step)

        steps_sorted = sorted(steps, key=lambda x: x["start"])

        if filepath.stem in sequences:
            logging.error("sequence already registered: this should not be reached")
        else:
            sequences[filepath.stem] = {
                "steps": steps_sorted,
                "with_error": flag_with_error,
            }

    return to_dict(sequences)


def get_sequences_from_coarse(dirpath):
    sequences = defaultdict(lambda: defaultdict(list))
    for filepath in dirpath.glob("*.txt"):
        with open(filepath, "r") as f:
            lines = f.readlines()

        steps = []
        for line in lines:
            start, end, action, *_ = line.split("\t")
            steps.append(
                {
                    "start": int(start) / 30,  # note: unit is now second
                    "end": int(end) / 30,
                    "action": action,
                }
            )

        sequence_id = filepath.stem.replace("disassembly_", "").replace("assembly_", "")
        sequences[sequence_id]["filenames"].append(filepath.stem)
        if filepath.name.startswith("assembly_"):
            steps_sorted = sorted(steps, key=lambda x: x["start"])
            sequences[sequence_id]["steps"] = steps_sorted

    for idx, example in sequences.items():
        if len(example["filenames"]) != 2:
            logging.debug(
                f"{idx} has only {[x.replace(idx, '') for x in example['filenames']]}"
            )

    return to_dict(sequences)


def find_longest_common_range(ranges):
    events = []
    for s, e in ranges:
        events.append((s, 1))
        events.append((e, -1))

    events.sort()

    active_count = 0
    start = None
    max_range = None
    max_length = 0

    for pos, event in events:
        active_count += event

        if active_count >= 2 and start is None:
            start = pos

        if active_count < 2 and start is not None:
            length = pos - start
            if length > max_length:
                max_length = length
                max_range = (start, pos)
            start = None

    return max_range


def get_sequences_from_fine(dirpath: Path):
    original_sequences = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    toy_id2name = {}
    splits = ["train.csv", "validation.csv"]
    for split in splits:
        logging.info(f"{split=}")
        df = pd.read_csv(dirpath / split)

        toy_id2name.update(df.set_index("toy_id")["toy_name"].to_dict())

        for row in tqdm(df.itertuples(index=False), total=len(df)):
            video_path = row.video.rsplit("/", 1)
            sequence_id = video_path[0] if len(video_path) > 1 else ""
            angle = video_path[-1].rsplit(".", 1)[0]
            key = f"{row.start_frame}-{row.end_frame}-{row.action_cls}"
            # note: unit is now second
            original_sequences[sequence_id][key]["start"] = int(row.start_frame) / 30
            original_sequences[sequence_id][key]["end"] = int(row.end_frame) / 30
            original_sequences[sequence_id][key]["action"] = row.action_cls
            original_sequences[sequence_id][key]["verb"] = row.verb_cls
            original_sequences[sequence_id][key]["noun"] = row.noun_cls
            original_sequences[sequence_id][key]["angles"].append(angle)

    sequences = defaultdict(lambda: defaultdict(list))
    for sequence_id, actions in original_sequences.items():
        # sort by start frame
        actions_sorted = sorted(actions.values(), key=lambda x: x["start"])

        # remove angle duplicates
        for action in actions_sorted:
            action["angles"] = sorted(list(set(action["angles"])))

        actions_sorted_overlap_collapsed = [[actions_sorted[0]]]
        for idx, action in enumerate(actions_sorted[1:]):
            prev_action = actions_sorted_overlap_collapsed[-1][-1]

            flag_overlap = False
            for prev_action in actions_sorted_overlap_collapsed[-1]:
                # if action description is identical and duration overlaps
                if prev_action["action"] == action["action"] and (
                    max(prev_action["start"], action["start"])
                    < min(prev_action["end"], action["end"])
                ):
                    flag_overlap = True

            if flag_overlap:
                # merge if prev_action['end'] == action['start']
                flag_ends_meet = False
                for prev_action in actions_sorted_overlap_collapsed[-1]:
                    if prev_action["end"] == action["start"]:
                        prev_action["end"] = action["end"]
                        flag_ends_meet = True
                        break

                if not flag_ends_meet:
                    actions_sorted_overlap_collapsed[-1].append(action)

                # sanity check
                if set(prev_action["angles"]) != set(action["angles"]):
                    logging.error("angle set is different: this does not happen")
            else:
                actions_sorted_overlap_collapsed.append([action])

        new_actions = []
        for same_actions in actions_sorted_overlap_collapsed:
            if len(same_actions) > 1:
                # identify common range with at least two ranges
                start, end = find_longest_common_range(
                    [[x["start"], x["end"]] for x in same_actions]
                )
                new_actions.append(
                    {
                        "start": start,
                        "end": end,
                        "action": same_actions[0]["action"],
                        "verb": same_actions[0]["verb"],
                        "noun": same_actions[0]["noun"],
                        "angle": same_actions[0]["angles"],
                    }
                )
            else:
                new_actions.append(dict(same_actions[0]))

        sequences[sequence_id]["steps"] = new_actions

    return to_dict(sequences)


def filter_and_merge(
    sequences_from_recording,
    sequences_from_mistake,
    sequences_from_coarse,
    sequences_from_fine,
):
    """
    filtering criteria:
    * sequences with all types of annotations
    * sequences with 12 views
    * sequences with both assembly and disassembly coarse annotation

    """
    intersect = list(
        set(sequences_from_recording.keys())
        & set(sequences_from_mistake.keys())
        & set(sequences_from_coarse.keys())
        & set(sequences_from_fine.keys())
    )
    sequences = defaultdict(dict)
    for sequence_id in sorted(intersect):
        if len(sequences_from_recording[sequence_id]["angles"]) != 12:
            continue
        if len(sequences_from_coarse[sequence_id]["filenames"]) != 2:
            continue

        sequences[sequence_id]["recording"] = sequences_from_recording[sequence_id]
        sequences[sequence_id]["mistake"] = sequences_from_mistake[sequence_id]
        sequences[sequence_id]["coarse"] = sequences_from_coarse[sequence_id]
        sequences[sequence_id]["fine"] = sequences_from_fine[sequence_id]

    return sequences


def format_caption(steps):
    caption = "WEBVTT\n\n"
    for step in steps:
        caption += f"{format_time(step['start'])} --> {format_time(step['end'])}\n"
        caption += f"{step['action']}"
        if step["label"] != "correct":
            caption += f" ({step['label']})"
        caption += "\n\n"

    return caption


def init_node(steps):
    nodes = [
        {
            "id": "0",
            "data": {"label": "start"},
            "position": {"x": 0, "y": 0},
        },
        {
            "id": "100",
            "data": {"label": "end"},
            "position": {"x": 0, "y": 800},
        },
    ]
    # -200, 0, 200
    for idx, step in enumerate(steps):
        x = -200 + (idx % 3) * 200
        y = 100 + (idx // 3) * 100
        nodes.append(
            {
                "id": f"{idx+1}",
                "type": "nodeWithScrew",
                "data": {"label": step, "checked": False},
                "position": {"x": x, "y": y},
            }
        )
    return nodes


def init_graph_annotation(sequences):
    toy_id2example = defaultdict(list)
    for sequence_id, sequence in sequences.items():
        toy_id = sequence["recording"]["toy_id"]
        steps = [
            x["action"]
            for x in sequence["mistake"]["steps"]
            if x["action"].startswith("attach")
        ]
        video = {
            "angles": sorted(sequence["recording"]["angles"]),
            "w-error": sequence["mistake"]["with_error"],
            "start": sequence["mistake"]["steps"][0]["start"],
            "caption": format_caption(sequence["mistake"]["steps"]),
        }

        if toy_id in toy_id2example:
            toy_id2example[toy_id]["all-steps"].extend(steps)
            toy_id2example[toy_id]["videos"][sequence_id] = video
        else:
            toy_id2example[toy_id] = {
                "all-steps": steps,
                "videos": {sequence_id: video},
            }

    examples = []
    for toy_id, example in sorted(toy_id2example.items(), key=lambda x: x[0]):
        steps = sorted(list(set(example["all-steps"])))
        examples.append(
            {
                "toy_id": toy_id,
                "nodes": init_node(steps),
                "edges": [],
                "filepath_image": f"{toy_id}-all.png",
                "videos": example["videos"],
            }
        )

    return examples


def update_mistake(sequences, filepath):
    """
    update mistake steps if necessary

    """

    with open(filepath, "r") as f:
        modifications = yaml.safe_load(f)

    for sequence_id, sequence in sequences.items():
        toy_id = sequence["recording"]["toy_id"]
        if toy_id not in modifications:
            continue
        modification = modifications[toy_id]
        for step in sequence["mistake"]["steps"]:
            if step["action"].startswith("attach"):
                if step["action"] in modification:
                    # update
                    step["action"] = modification[step["action"]]
            elif step["action"].startswith("detach"):
                attach_version = (
                    step["action"].replace("detach", "attach").replace(" from ", " to ")
                )
                if attach_version in modification:
                    attach_version_updated = modification[attach_version]
                    original_updated = attach_version_updated.replace(
                        "attach", "detach"
                    ).replace(" to ", " from ")
                    step["action"] = original_updated
    return sequences


def main(args):
    logging.info("### Check recording files ###")
    sequences_from_recording = get_sequences_from_recording(args.dirpath_recording)
    get_stats(sequences_from_recording, ["recording"])

    logging.info("### Check mistake annotation ###")
    sequences_from_mistake = get_sequences_from_mistake(args.dirpath_mistake)
    get_stats(sequences_from_mistake, ["mistake"])

    logging.info("### Check coarse annotation ###")
    sequences_from_coarse = get_sequences_from_coarse(args.dirpath_coarse)
    get_stats(sequences_from_coarse, ["coarse"])

    logging.info("### Check fine annotation ###")
    sequences_from_fine = get_sequences_from_fine(args.dirpath_fine)
    get_stats(sequences_from_fine, ["fine"])

    logging.info("### Filtering & Merge ###")
    sequences_merged_filtered = filter_and_merge(
        sequences_from_recording,
        sequences_from_mistake,
        sequences_from_coarse,
        sequences_from_fine,
    )
    get_stats(sequences_merged_filtered, ["all"])

    logging.info("### Update ###")
    sequences_merged_filtered = update_mistake(
        sequences_merged_filtered, args.filepath_modification
    )

    logging.info("### Output examples ###")
    output = {
        "metadata": {
            "data-created": datetime.today().strftime("%Y-%m-%d"),
            "total": len(sequences_merged_filtered),
        },
        "examples": to_dict(sequences_merged_filtered),
    }
    with open(args.dirpath_output / "all_in_one.json", "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")

    logging.info("### Init graph annotation ###")
    graphs = init_graph_annotation(sequences_merged_filtered)

    logging.info("### Output graphs ###")
    dirpath_output_graph = args.dirpath_output / "graph"
    if not dirpath_output_graph.exists():
        dirpath_output_graph.mkdir(parents=True)

    toy_ids = list([x["toy_id"] for x in graphs])
    output = {
        "metadata": {
            "data-created": datetime.today().strftime("%Y-%m-%d"),
            "annotator_id": "all",
            "total": len(graphs),
            "toy_ids": toy_ids,
        },
        "examples": graphs,
    }
    with open(dirpath_output_graph / "all.json", "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Format Assembly101 annotation")
    parser.add_argument("--dirpath_recording", type=Path, help="dirpath to recording")
    parser.add_argument(
        "--dirpath_mistake", type=Path, help="dirpath to mistake annotation"
    )
    parser.add_argument(
        "--dirpath_coarse", type=Path, help="dirpath to coarse action annotation"
    )
    parser.add_argument(
        "--dirpath_fine", type=Path, help="dirpath to fine action annotation"
    )
    parser.add_argument(
        "--filepath_modification",
        type=Path,
        help="filepath for annotation modification",
    )
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument("--dirpath_log", type=Path, help="log")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir(parents=True)

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "format_annotation.log"),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
