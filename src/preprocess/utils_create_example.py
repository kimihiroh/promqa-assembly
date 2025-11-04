"""
util functions for create_example

"""

from collections import defaultdict
import networkx as nx
from copy import deepcopy
import re
import logging


def format_metadata(sequence_id, annotation):
    output = {
        "user_id": annotation["recording"]["user_id"],
        "toy_id": annotation["recording"]["toy_id"],
        "recording_id": annotation["recording"]["recording_id"],
        "sequence_id": sequence_id,
        "start": annotation["mistake"]["steps"][0]["start"],
    }
    return output


def create_init_step():
    output = {
        "step_id": 0,
        "action": "start",
        "start": 0,
        "end": 0,
        "type": None,
        "mistake": None,
    }  # initial step
    return output


def format_current_step(idx, step):
    output = {
        "step_id": idx + 1,
        "action": step["action"],
        "start": step["start"],
        "end": step["end"],
        "type": step["label"],
        "mistake": step["remark"] if step["remark"] else None,
    }
    return output


def check_overlap_fine_coarse(target_fine_step, coarse_steps):
    outputs = []
    for idx, coarse_step in enumerate(coarse_steps):
        overlap_start = max(target_fine_step["start"], coarse_step["start"])
        overlap_end = min(target_fine_step["end"], coarse_step["end"])
        if overlap_start < overlap_end:
            outputs.append([idx, overlap_end - overlap_start])

    output = None
    if len(outputs) == 1:
        output = outputs[0][0]
    elif len(outputs) > 1:
        min_output = outputs[0]
        for _output in outputs[1:]:
            if _output[1] > min_output[1]:
                min_output = _output
        output = min_output[0]
    else:
        pass

    return output


def get_all_fine_steps(annotation):
    """
    group fine steps to each corresponding coarse step
    """
    recording_start = annotation["mistake"]["steps"][0]["start"]
    recording_end = annotation["mistake"]["steps"][-1]["end"]

    centers_mistake = [
        (x["start"] + x["end"]) / 2 for x in annotation["mistake"]["steps"]
    ]

    mistake2fine = defaultdict(list)
    for id_fine, step_fine in enumerate(annotation["fine"]["steps"]):
        # skip fine steps when before or after main procedure
        if step_fine["end"] <= recording_start or recording_end < step_fine["start"]:
            continue

        # if overlap, use it
        corresponding_id = check_overlap_fine_coarse(
            step_fine, annotation["mistake"]["steps"]
        )

        # if no overlap exists, choose based on the distance with centers
        center_fine = (step_fine["start"] + step_fine["end"]) / 2
        if not corresponding_id:
            distances = [abs(x - center_fine) for x in centers_mistake]
            min_distance = min(distances)
            indices_min_mistake = [
                idx for idx, x in enumerate(distances) if x == min_distance
            ]

            if len(indices_min_mistake) == 1:
                corresponding_id = indices_min_mistake[0]
            else:
                # hard-coded: choose the first one, only two cases
                corresponding_id = indices_min_mistake[0]
                # print(f'hey: {id_fine} {indices_min_mistake}')

        mistake2fine[corresponding_id + 1].append(id_fine)
    return mistake2fine


def check_if_noisy(previous_steps):
    """check if mistakes and/or corrections exist"""
    flag = False
    for step in previous_steps:
        if step["mistake"]:
            flag = True
            break
    return flag


def estimate_screw_state(current_step, mapping_coarse_to_fine, fine_steps, graph):
    """
    estimate screw state for each prev+current steps

    limitation:
    only from the current step range
    if the screw for the current step is done during the following steps,
    it will be ignored.

    todo: remove if not needed
    """
    G = graph["graph"]
    id2action = graph["id2action"]
    id2screw = graph["id2screw"]
    target_actions = {
        id2action[node]: id2screw[node] for node in G.nodes if G.degree[node] > 0
    }

    flag_screw = None
    if (
        current_step["action"] in target_actions
        and target_actions[current_step["action"]]
    ):
        fine_steps_in_range = [
            fine_steps[x] for x in mapping_coarse_to_fine[current_step["step_id"]]
        ]
        flag_screw = any([x["action"].startswith("screw") for x in fine_steps_in_range])

    return flag_screw


def format_example_base(idx, sequence_id, annotation, previous_steps, graph):
    metadata = format_metadata(sequence_id, annotation)
    step = annotation["mistake"]["steps"][idx]
    mapping_coarse_to_fine = get_all_fine_steps(annotation)

    current_step = format_current_step(idx, step)
    current_step["w_screw"] = (
        graph["id2screw"][graph["action2id"][current_step["action"]]]
        if current_step["action"] in graph["action2id"]
        else None
    )
    current_step["is_screwed"] = estimate_screw_state(
        current_step,
        mapping_coarse_to_fine,
        annotation["fine"]["steps"],
        graph,
    )

    example_base = {
        "metadata": metadata,
        "end": step["end"],
        "previous_steps": deepcopy(previous_steps),
        "current_step": deepcopy(current_step),
        "is_noisy": check_if_noisy(previous_steps),
        "mapping_coarse_to_fine": mapping_coarse_to_fine,
    }

    # update previous steps list
    previous_steps.append(deepcopy(current_step))

    return example_base, previous_steps


def check_overlap_coarse(idx, steps):
    """
    note:
    * this is not true for a101
    """

    current_step = steps[idx]
    start, end = current_step["start"], current_step["end"]

    flag = False
    for step in steps[idx + 1 :]:
        _start, _end = step["start"], step["end"]
        if max(start, _start) < min(end, _end):
            flag = True

    if flag:
        logging.warning(f"Overlap: {idx=}")

    return flag


def convert_action_to_detach(action):
    """detach X from Y -> attach X to Y"""
    match = re.search(r"detach (.+) from (.+)", action)
    return f"attach {match.group(1)} to {match.group(2)}"


def get_passed_steps(toy_id, steps):
    """
    attach X to Y -> detach X from Y => cancel.
    """
    passed_steps_label = []
    for step in steps:
        if step["action"].startswith("detach"):
            idx = None
            corresponding_action = convert_action_to_detach(step["action"])
            for _idx, _step in enumerate(passed_steps_label):
                if corresponding_action == _step:
                    idx = _idx
                    # no break to find the latest one
            if idx:
                passed_steps_label.pop(idx)
        else:
            passed_steps_label.append(step["action"])

    return passed_steps_label


def get_remaining_parts(graph, example_base):
    """
    get remaining parts

    todo:
    * better handle detach cases

    """
    G = graph["graph"]
    id2action = graph["id2action"]

    passed_steps_label = get_passed_steps(
        graph["toy_id"], example_base["previous_steps"] + [example_base["current_step"]]
    )
    nodes_with_edges = [node for node in G.nodes if G.degree[node] > 0]
    remaining_parts = []
    for node in nodes_with_edges:
        if node in ["0", "100"]:  # skip for start and end node
            continue
        if id2action[node] not in passed_steps_label:
            match = re.search(r"attach (.*?) to", id2action[node])
            if match:
                remaining_parts.append(match.group(1))

    return sorted(list(set(remaining_parts)))


def get_next_steps(graph, example_base):
    """get next steps"""
    G = graph["graph"]
    id2action = graph["id2action"]

    passed_steps_label = get_passed_steps(
        graph["toy_id"], example_base["previous_steps"] + [example_base["current_step"]]
    )
    next_steps = []
    # only consider nodes that are connected
    nodes_with_edges = [node for node in G.nodes if G.degree[node] > 0]
    for node in nodes_with_edges:
        if node in ["0", "100"]:  # skip for start and end node
            continue
        if id2action[node] not in passed_steps_label:
            # check if all predecessors of this node are in passes_steps
            all_predecessors_passed = all(
                id2action[pred] in passed_steps_label for pred in G.predecessors(node)
            )

            # check if no succeeding step is passed
            # otherwise missing steps would be added as next steps
            succeeding_step_found = any(
                id2action[pred] in passed_steps_label
                for pred in nx.descendants(G, node)
            )
            if all_predecessors_passed and not succeeding_step_found:
                next_steps.append(id2action[node])

    return next_steps


def format_example_next(graph, example_base):
    return example_base | {
        "type": "next",
        "next_steps": get_next_steps(graph, example_base),
        "remaining_parts": get_remaining_parts(graph, example_base),
    }


def get_missing_steps(graph, example_base):
    """
    get missing steps

    missing step is a step where at least
    one of preceding steps and one of succeeddings are performed

    note: maybe the condition, onc preceding step, may not be required
    """

    G = graph["graph"]
    id2action = graph["id2action"]

    passed_steps_label = get_passed_steps(
        graph["toy_id"], example_base["previous_steps"] + [example_base["current_step"]]
    )
    missing_steps = []
    # only consider nodes that are connected
    nodes_with_edges = [node for node in G.nodes if G.degree[node] > 0]
    for node in nodes_with_edges:
        if node in ["0", "100"]:  # skip for start and end node
            continue
        if id2action[node] not in passed_steps_label:
            # at least one preceding step is passed
            preceding_step_found = any(
                id2action[pred] in passed_steps_label for pred in nx.ancestors(G, node)
            )
            # at least one succeeding is passed
            succeeding_step_found = any(
                id2action[pred] in passed_steps_label
                for pred in nx.descendants(G, node)
            )
            if preceding_step_found and succeeding_step_found:
                missing_steps.append(id2action[node])
    return missing_steps


def format_example_missing(graph, example_base):
    """
    target: missing based on graph

    note:
    * missing based on graph is identifying cases
      where detachment was incomplete

    """
    missing_steps = get_missing_steps(graph, example_base)
    return example_base | {
        "type": "missing",
        "missing_steps": missing_steps,
        "remaining_parts": get_remaining_parts(graph, example_base),
    }


def format_example_order(graph, example_base):
    """
    target: "wrong order"

    """
    flag = example_base["current_step"]["mistake"] == "wrong order"
    return example_base | {
        "type": "order",
        "wrong_order_annotation": flag,
        "remaining_parts": get_remaining_parts(graph, example_base),
    }


def format_example_past(graph, example_base):
    """
    target: the latest mistake step or None
    """
    flag = example_base["current_step"]["mistake"] == "previous one is a mistake"
    return example_base | {
        "type": "past",
        "error_accumulation": flag,
        "remaining_parts": get_remaining_parts(graph, example_base),
    }


def format_example_misadjustment(graph, example_base):
    flag = example_base["current_step"]["mistake"] == "shouldn't have happened"
    return example_base | {
        "type": "misadjustment",
        "misadjustment": flag,
        "remaining_parts": get_remaining_parts(graph, example_base),
    }


def format_example_general(graph, example_base):
    """for general question about any of prev&current ones"""
    return example_base | {
        "type": "general",
        "remaining_parts": get_remaining_parts(graph, example_base),
    }


def format_example_current_general(graph, example_base):
    """for general question about current one"""
    return example_base | {
        "type": "current_general",
    }


def format_example_location(graph, example_base):
    flag = example_base["current_step"]["mistake"] == "wrong position"
    return example_base | {
        "type": "location",
        "location_error": flag,
    }


def create_graph(annotation):
    G = nx.DiGraph()
    id2step, step2id, id2screw = {}, {}, {}
    for x in annotation["nodes"]:
        id2step[x["id"]] = x["data"]["label"]
        step2id[x["data"]["label"]] = x["id"]
        if "checked" in x["data"]:
            id2screw[x["id"]] = x["data"]["checked"]
        else:
            id2screw[x["id"]] = False
    for edge in annotation["edges"]:
        G.add_edge(edge["source"], edge["target"])
    output = {
        "toy_id": annotation["toy_id"],
        "graph": G,
        "id2action": id2step,
        "action2id": step2id,
        "id2screw": id2screw,
    }

    return output
