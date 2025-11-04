"""
Download data from HF Hub

"""

from argparse import ArgumentParser
from datasets import load_dataset
import logging
from pathlib import Path
from pprint import pformat
from utils import (
    SUBSETS,
)
from src.utils.misc import (
    get_date,
)


def main(args):
    if args.target_data in ["cookiing", "both"]:
        pass

    if args.target_data in ["assembly", "both"]:
        for toy_id in SUBSETS:
            load_dataset("kimihiroh/promqa-assembly", toy_id, split="test")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download data")
    parser.add_argument(
        "--target",
        type=str,
        help="target data to download",
        choices=["cooking", "assembly", "both"],
        default="both",
    )
    parser.add_argument(
        "--dirpath_log", type=Path, help="dirpath for log", default="./log"
    )
    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir(parents=True)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / f"download_{get_date()}.log"),
        ],
    )

    logging.info(pformat(vars(args)))

    main(args)
