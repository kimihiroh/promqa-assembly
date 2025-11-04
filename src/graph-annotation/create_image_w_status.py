"""
create graphs w/ status

"""

from argparse import ArgumentParser
from collections import defaultdict
import emoji
import json
import logging
from pathlib import Path
import pydot
import textwrap
from tqdm import tqdm


def create_graph_w_status(example, graph):
    """
    Create task graphs w/ status
    """

    # todo
    # toy_id = graph["toy_id"]
    edges = graph["edges"]
    valid_node_ids = []
    for edge in edges:
        valid_node_ids.append(edge["source"])
        valid_node_ids.append(edge["target"])
    valid_node_ids = list(set(valid_node_ids))

    assert len(valid_node_ids) == len(graph["nodes"])

    node_id2text = {}
    node_id2screw = {}
    for node in graph["nodes"]:
        node_id2text[node["id"]] = node["data"]["label"]
        if "checked" in node["data"] and node["data"]["checked"]:
            node_id2screw[node["id"]] = True
        else:
            node_id2screw[node["id"]] = False

    node_id2status = defaultdict(list)
    node_id_current_step = None
    for idx, step in enumerate(example["previous_steps"] + [example["current_step"]]):
        # find corresponding node_id
        ## convert detach action to attach action
        if step["action"].startswith("detach"):
            action = (
                step["action"].replace("detach", "attach").replace(" from ", " to ")
            )
        else:
            action = step["action"]

        corresponding_id = None
        for node in graph["nodes"]:
            if node["data"]["label"] == action:
                corresponding_id = node["id"]
        if not corresponding_id:
            # some nodes do not have a corresponding node in graph
            # simply because the action is wrong
            logging.debug(
                f"{example['metadata']['toy_id']} {example['metadata']['recording_id']}: "
                f"{action} does not have a corresponding node in task graph"
            )

        # status
        status = ""
        if step["action"].startswith("detach"):
            status = f"{step['step_id']} {emoji.emojize(':wrench:')}"
        else:
            match step["type"]:
                case "correct":
                    status = f"{step['step_id']} {emoji.emojize(':check_mark_button:')}"
                case "correction":
                    status = f"{step['step_id']} {emoji.emojize(':wrench:')}"
                case "mistake":
                    status = f"{step['step_id']} {emoji.emojize(':warning:')}"
                case None:
                    status = ""
                case _:
                    logging.error(f"Undefined {step['type']=}")

        node_id2status[corresponding_id].append(status)

        # to distinguish current step with others
        if idx == len(example["previous_steps"]):
            node_id_current_step = corresponding_id

    G = pydot.Dot(graph_type="digraph")
    id2node = {}
    for node_id in valid_node_ids:
        node_text = node_id2text[node_id]
        if node_id2screw[node_id]:
            node_text = f"{emoji.emojize(':nut_and_bolt:')} {node_text}"
        node_text = textwrap.fill(node_text, width=20)

        # check its status
        if node_id in node_id2status:
            status = ", ".join(node_id2status[node_id])
            status = textwrap.fill(status, width=20)
            if node_id == node_id_current_step:
                node = pydot.Node(
                    f"{node_text}\n{status}",
                    shape="box",
                    style="diagonals, bold",
                )
            else:
                node = pydot.Node(
                    f"{node_text}\n{status}",
                    shape="box",
                    style="solid",
                )
        else:
            node = pydot.Node(
                f"{node_text}",
                shape="box",
                style="dotted",
            )

        id2node[node_id] = node
        G.add_node(node)

    for edge in edges:
        edge = pydot.Edge(
            id2node[edge["source"]],
            id2node[edge["target"]],
        )
        G.add_edge(edge)

    return G


def main(args):
    # load input
    with open(args.filepath_input, "r") as f:
        data = json.load(f)

    # load graph
    with open(args.filepath_graph, "r") as f:
        graphs = json.load(f)
    id2graph = {x["toy_id"]: x for x in graphs["examples"]}

    for example in tqdm(data["examples"]):
        filename = (
            f"{example['metadata']['user_id']}"
            f"-{example['metadata']['toy_id']}"
            f"-{example['metadata']['recording_id']}"
            f"-{example['current_step']['step_id']}.png"
        )
        filepath = args.dirpath_output / filename

        if filepath.exists():  # skip if exists
            continue

        G = create_graph_w_status(example, id2graph[example["metadata"]["toy_id"]])
        G.write_png(filepath)


if __name__ == "__main__":
    parser = ArgumentParser(description="create instruction w/ status png files")
    parser.add_argument("--filepath_input", type=Path, help="filepath to input data")
    parser.add_argument("--filepath_graph", type=Path, help="filepath to graphs")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "instruction_w_status_a101.log"),
        ],
    )
    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir()

    logging.info(f"Arguments: {vars(args)}")

    main(args)
