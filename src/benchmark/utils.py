"""
helper functions

"""

import logging
import pydot
import re
import json


SUBSETS = [
    'a02', 'a03', 'a07', 'a08', 'a09', 'a10', 'a12', 'a13', 'a14', 'a16', 'a18', 'a19', 
    'a20', 'a23', 'a24', 'a26', 'a28', 'a29', 'a30', 'b01a', 'b01b', 'b02a', 'b02b', 
    'b03a', 'b03b', 'b04a', 'b04b', 'b04c', 'b04d', 'b05a', 'b05b', 'b05c', 'b05d', 
    'b06a', 'b06b', 'b06c', 'b06d', 'b08a', 'b08b', 'b08d', 'c02a', 'c02b', 'c02c', 
    'c03a', 'c03b', 'c03c', 'c03d', 'c03e', 'c03f', 'c04a', 'c04d', 'c05a', 'c06a', 
    'c06b', 'c06c', 'c06d', 'c06f', 'c07a', 'c07b', 'c07c', 'c08a', 'c08b', 'c08c', 
    'c09a', 'c09b', 'c09c', 'c10a', 'c10b', 'c10c', 'c11a', 'c12a', 'c12e', 'c13a', 
    'c13c', 'c13d', 'c13e', 'c13f', 'c14a'
]


def format_instruction():
    logging.info("Load instructions ... ")
    
    graphs = load_dataset("kimihiroh/assembly101-graph", split="test")
    
    
    toy2instruction = {}
    for graph in graphs:
        
        toy_id = graph["toy_id"]

        G = pydot.Dot(graph_type="digraph")
        action_id2description = {}
        for action in graph["nodes"]:
            idx = str(action["id"])
            if "checked" in action and action["checked"]:
                description = f"{action['label']} w/ screw"
            else:
                description = f"{action['label']}"
            action_id2description[idx] = description
            node = pydot.Node(f"{description}")
            G.add_node(node)

        for edge in graph["edges"]:
            edge = pydot.Edge(
                action_id2description[str(edge["source"])],
                action_id2description[str(edge["target"])],
            )
            G.add_edge(edge)

        toy2instruction[toy_id] = {
            "dot": G.to_string().strip(),
            "dag": graph['graph'],
            "parts": f'./data/parts/{toy_id}-all.png'
        }

    return toy2instruction


def extract_index(filepath):
    return int(re.search(r"\d+", filepath.stem).group())


def sample_frame(start, end, max_frames):
    target_filepaths = []
    for idx in range(int(start), int(end)):
        target_filepaths.append(idx)

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


# def format_steps(steps: list, w_error: bool = False) -> str:
#     output = ""
#     for step in steps:
#         output += f"- {step['description']}\n"
#         if w_error and "errors" in step:
#             for error in step["errors"]:
#                 output += f"    - [{error['tag']}] {error['description']}\n"

#     return output.strip()

