"""
prepare for verification
* format examples
* create graphs

"""

from argparse import ArgumentParser
import emoji
import json
import logging
from pathlib import Path
import pydot
import textwrap
from tqdm import tqdm
import subprocess


def create_graph(graph, dirpath):
    """
    create one plain graph as image for each toy

    """
    # create graph
    filepath = dirpath / f"{graph['toy_id']}.png"

    edges = graph["edges"]
    valid_node_ids = []
    for edge in edges:
        valid_node_ids.append(edge["source"])
        valid_node_ids.append(edge["target"])

    id2text = {}
    id2screw = {}
    for node in graph["nodes"]:
        id2text[node["id"]] = node["data"]["label"]
        if "checked" in node["data"] and node["data"]["checked"]:
            id2screw[node["id"]] = True
        else:
            id2screw[node["id"]] = False

    G = pydot.Dot(graph_type="digraph")
    id2node = {}
    for node_id in valid_node_ids:
        node_text = id2text[node_id]
        if id2screw[node_id]:
            node_text = f"{emoji.emojize(':nut_and_bolt:')} {node_text}"
        node_text = textwrap.fill(node_text, width=20)

        node = pydot.Node(
            f"{node_text}",
            shape="box",
            style="solid",
        )

        id2node[node_id] = node
        G.add_node(node)

    for edge in edges:
        edge = pydot.Edge(
            id2node[edge["source"]],
            id2node[edge["target"]],
        )
        G.add_edge(edge)

    # output graph as file
    G.write_png(filepath)

    # return filename
    return str(filepath.name)


def main(args):
    # load graph
    with open(args.filepath_input, "r") as f:
        graphs = json.load(f)

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    for graph in tqdm(graphs["examples"]):
        create_graph(graph, args.dirpath_output)

    dirpath_parts_output = args.dirpath_parts / "360p"
    if not dirpath_parts_output.exists():
        dirpath_parts_output.mkdir(parents=True)

    for graph in tqdm(graphs["examples"]):
        filepath = args.dirpath_parts / "original" / f"{graph['toy_id']}-all.png"
        filepath_output = dirpath_parts_output / f"{graph['toy_id']}-all.png"
        command = [
            "ffmpeg",
            "-i",
            str(filepath),
            "-vf",
            "scale=-1:360",
            str(filepath_output),
        ]
        subprocess.run(command)


if __name__ == "__main__":
    parser = ArgumentParser(description="Sample examples")
    parser.add_argument("--filepath_input", type=Path, help="filepath to input data")
    parser.add_argument("--dirpath_parts", type=Path, help="dirpath to parts images")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir(parents=True)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "create_image.log"),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
