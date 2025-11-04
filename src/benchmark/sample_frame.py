"""
Preprocess videos and sample frames

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from tqdm import tqdm
import subprocess
import multiprocessing
from collections import defaultdict

CONFIGS_FRAMES = {
    "rgb": [
        ["640", "360"],
    ],
    "wb": [],
}

CONFIGS_SAMPLING = {
    "fps": 1,
    "rgb": ["1920", "1080"],
    "wb": ["636", "480"],
}


def format_time(seconds: float):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    output = f"{int(h):02}:{int(m):02}:{int(s):02}"
    return output


def worker_function(command):
    status = 0
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        if result.returncode != 0 and status == 0:
            status = result.returncode
    except Exception as e:
        logging.error(f"In worker_function(): {e}")
        if status == 0:
            status = 1

    return status


def create_command_frame_sampling(filepath, start_second, dirpath_output, video_type):
    fps = CONFIGS_SAMPLING["fps"]
    width, height = CONFIGS_SAMPLING[video_type]
    filepath_output = (
        dirpath_output
        / video_type
        / f"{height}p"
        / filepath.parent.name
        / filepath.stem
        / "%d.png"
    )
    if not filepath_output.parent.exists():
        filepath_output.parent.mkdir(parents=True)

    start_format = format_time(start_second)

    # note: option order matters
    command = [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",
        "-ss",
        start_format,
        "-i",
        str(filepath),
        "-vf",
        f"fps={fps}",
        "-start_number",
        f"{int(start_second)}",
        filepath_output,
    ]

    return command


def sample_frames(dirpath_input, annotations, dirpath_output, max_parallel_jobs):
    # frame sampling
    if not dirpath_output.exists():
        dirpath_output.mkdir(parents=True)

    commands = []
    count = defaultdict(int)
    for folderpath in args.dirpath_input.glob("nusar-2021*"):
        if folderpath.stem not in annotations["examples"]:
            count["skipped"] += 1
            continue

        count["total"] += 1

        for filepath in folderpath.glob("*.mp4"):
            if filepath.name.startswith("C10"):
                video_type = "rgb"
            elif filepath.name.startswith("HMC"):
                video_type = "wb"
            else:
                logging.error(f"Undefined video type: {filepath.name}")

            annotation = annotations["examples"][folderpath.stem]
            start_second = annotation["mistake"]["steps"][0]["start"]

            command = create_command_frame_sampling(
                filepath, start_second, dirpath_output, video_type
            )
            commands.append(command)

    logging.info(f"#command: {len(commands)}")
    logging.info(f"#recording: {dict(count)}")

    results = []
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker_function, commands), total=len(commands))
        )

    count = defaultdict(int)
    for status in results:
        if status == 0:
            count["success"] += 1
        else:
            count["failure"] += 1
    logging.info(f"[count] success: {count['success']}, failure: {count['failure']}")

    return count


def create_command_frame_resolution(filepath, dirpath_output, video_type):
    commands = []
    for width, height in CONFIGS_FRAMES[video_type]:
        filepath_output = (
            dirpath_output
            / video_type
            / f"{height}p"
            / filepath.parent.parent.name
            / filepath.parent.name
            / filepath.name
        )
        if not filepath_output.parent.exists():
            filepath_output.parent.mkdir(parents=True)

        # kh: option order matters
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(filepath),
            "-vf",
            f"scale={width}:{height}",
            str(filepath_output),
        ]
        commands.append(command)

    return commands


def change_frame_resolution(dirpath_input, dirpath_output, max_parallel_jobs):
    logging.info("Create frames with different resolutions ...")

    if not dirpath_output.exists():
        dirpath_output.mkdir(parents=True)

    commands = []
    for folderpath in (dirpath_input / "rgb" / "1080p").glob("*"):
        for filepath in folderpath.glob("*/*.png"):
            command = create_command_frame_resolution(filepath, dirpath_output, "rgb")
            commands.extend(command)

    logging.info(f"#command: {len(commands)}")

    results = []
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        results = list(
            tqdm(pool.imap_unordered(worker_function, commands), total=len(commands))
        )

    count = defaultdict(int)
    for status in results:
        if status == 0:
            count["success"] += 1
        else:
            count["failure"] += 1
    logging.info(f"[count] success: {count['success']}, failure: {count['failure']}")

    return count


def main(args):
    """
    sample video frames and change resolution for benchmarking
    - sample: 1 fps
    - resolution: 360p (for RGB. no change for WB)

    Only for the assembling part (no sampling from disassembling part)
    """

    with open(args.filepath_annotation, "r") as f:
        annotations = json.load(f)

    sample_frames(
        args.dirpath_input,
        annotations,
        args.dirpath_output / "frames",
        args.max_parallel_jobs,
    )

    change_frame_resolution(
        args.dirpath_output / "frames",
        args.dirpath_output / "frames",
        args.max_parallel_jobs,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess videos")
    parser.add_argument("--dirpath_input", type=Path, help="dirpath to input")
    parser.add_argument(
        "--filepath_annotation", type=Path, help="filepath to annotation"
    )
    parser.add_argument("--dirpath_output", type=Path, help="dirpath_output")
    parser.add_argument(
        "--max_parallel_jobs", type=int, help="max #parallel jobs", default=1
    )
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "preprocess_video.log"),
        ],
    )

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir(parents=True)
    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    logging.info(f"Arguments: {vars(args)}")

    main(args)
