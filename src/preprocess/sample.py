"""
Sample examples for Assembly101

"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import json
from collections import defaultdict
import random
from copy import deepcopy
from datetime import datetime

from create_example import (
    stats,
    QUESTION_TYPES,
)


def get_group_size(num_total_sample):
    type2num = {}
    default_size = num_total_sample // len(QUESTION_TYPES)
    remainder = num_total_sample % len(QUESTION_TYPES)

    # distribute evenly, with reminder goes to a few initial groups
    for idx, _type in enumerate(QUESTION_TYPES):
        size = default_size + (1 if idx < remainder else 0)
        type2num[_type] = size

    return type2num


def sample(_type, id2examples, num):
    samples, remainings = [], []
    match _type:
        case "next":
            """randomly pick one example from each recording until num"""
            ids = list(id2examples.keys())
            random.shuffle(ids)
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                if i < num:
                    random.shuffle(_examples)
                    samples.append(_examples[0])
                    remainings += _examples[1:]
                else:
                    remainings += _examples
        case "missing":
            """
            at most one pos and neg from each recording
            select at most num//2 pos, and neg will be remaining
            """

            # categorization based on recording id and pos/neg
            ids = list(id2examples.keys())
            random.shuffle(ids)
            pos_samples, neg_samples = [], []
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                _pos_examples, _neg_examples = [], []
                for _example in _examples:
                    # categorize
                    if len(_example["missing_steps"]) > 0:
                        _pos_examples.append(_example)
                    else:
                        _neg_examples.append(_example)

                if len(_pos_examples) > 0:
                    random.shuffle(_pos_examples)
                    pos_samples.append(_pos_examples[0])
                    remainings += _pos_examples[1:]

                if len(_neg_examples) > 0:
                    random.shuffle(_neg_examples)
                    neg_samples.append(_neg_examples[0])
                    remainings += _neg_examples[1:]

            # at most num//2 pos and neg for remaining
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)
            if num % 2 == 0:
                pos_size, neg_size = num // 2, num // 2
            else:
                pos_size, neg_size = num // 2 + 1, num // 2

            # adjust the sizes
            if len(pos_samples) < pos_size:
                diff = pos_size - len(pos_samples)
                pos_size -= diff
                neg_size += diff

            samples += pos_samples[:pos_size]
            remainings += pos_samples[pos_size:]
            samples += neg_samples[:neg_size]
            remainings += neg_samples[neg_size:]
        case "order":
            """
            at most one pos and neg from each recording
            select at most num//2 pos, and neg will be remaining
            """

            # categorization based on recording id and pos/neg
            ids = list(id2examples.keys())
            random.shuffle(ids)
            pos_samples, neg_samples = [], []
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                _pos_examples, _neg_examples = [], []
                for _example in _examples:
                    # categorize
                    if _example["wrong_order_annotation"]:
                        _pos_examples.append(_example)
                    else:
                        _neg_examples.append(_example)

                if len(_pos_examples) > 0:
                    random.shuffle(_pos_examples)
                    pos_samples.append(_pos_examples[0])
                    remainings += _pos_examples[1:]

                if len(_neg_examples) > 0:
                    random.shuffle(_neg_examples)
                    neg_samples.append(_neg_examples[0])
                    remainings += _neg_examples[1:]

            # at most num//2 pos and neg for remaining
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)
            if num % 2 == 0:
                pos_size, neg_size = num // 2, num // 2
            else:
                pos_size, neg_size = num // 2 + 1, num // 2

            # adjust the sizes
            if len(pos_samples) < pos_size:
                diff = pos_size - len(pos_samples)
                pos_size -= diff
                neg_size += diff

            samples += pos_samples[:pos_size]
            remainings += pos_samples[pos_size:]
            samples += neg_samples[:neg_size]
            remainings += neg_samples[neg_size:]
        case "past":
            """
            at most one pos and neg from each recording
            select at most num//2 pos, and neg will be remaining
            """

            # categorization based on recording id and pos/neg
            ids = list(id2examples.keys())
            random.shuffle(ids)
            pos_samples, neg_samples = [], []
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                _pos_examples, _neg_examples = [], []
                for _example in _examples:
                    # categorize
                    if bool(_example["error_accumulation"]):
                        _pos_examples.append(_example)
                    else:
                        _neg_examples.append(_example)

                if len(_pos_examples) > 0:
                    random.shuffle(_pos_examples)
                    pos_samples.append(_pos_examples[0])
                    remainings += _pos_examples[1:]

                if len(_neg_examples) > 0:
                    random.shuffle(_neg_examples)
                    neg_samples.append(_neg_examples[0])
                    remainings += _neg_examples[1:]

            # at most num//2 pos and neg for remaining
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)
            if num % 2 == 0:
                pos_size, neg_size = num // 2, num // 2
            else:
                pos_size, neg_size = num // 2 + 1, num // 2

            # adjust the sizes
            if len(pos_samples) < pos_size:
                diff = pos_size - len(pos_samples)
                pos_size -= diff
                neg_size += diff

            samples += pos_samples[:pos_size]
            remainings += pos_samples[pos_size:]
            samples += neg_samples[:neg_size]
            remainings += neg_samples[neg_size:]

        case "misadjustment":
            """only pos"""

            # categorization based on recording id and pos/neg
            ids = list(id2examples.keys())
            random.shuffle(ids)
            pos_samples = []
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                _pos_examples = []
                for _example in _examples:
                    # categorize
                    if _example["misadjustment"]:
                        _pos_examples.append(_example)
                    else:
                        remainings.append(_example)

                if len(_pos_examples) > 0:
                    random.shuffle(_pos_examples)
                    pos_samples.append(_pos_examples[0])
                    remainings += _pos_examples[1:]

            # add at most num
            random.shuffle(pos_samples)
            samples += pos_samples[:num]
            remainings += pos_samples[num:]

        case "general":
            ids = list(id2examples.keys())
            random.shuffle(ids)
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                if i < num:
                    random.shuffle(_examples)
                    samples.append(_examples[0])
                    remainings += _examples[1:]
                else:
                    remainings += _examples
        case "current_general":
            ids = list(id2examples.keys())
            random.shuffle(ids)
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                if i < num:
                    random.shuffle(_examples)
                    samples.append(_examples[0])
                    remainings += _examples[1:]
                else:
                    remainings += _examples
        case "location":
            """
            at most one pos and neg from each recording
            select at most num//2 pos, and neg will be remaining
            """

            # categorization based on recording id and pos/neg
            ids = list(id2examples.keys())
            random.shuffle(ids)
            pos_samples, neg_samples = [], []
            for i, idx in enumerate(ids):
                _examples = id2examples[idx]
                _pos_examples, _neg_examples = [], []
                for _example in _examples:
                    # categorize
                    if bool(_example["location_error"]):
                        _pos_examples.append(_example)
                    else:
                        _neg_examples.append(_example)

                if len(_pos_examples) > 0:
                    random.shuffle(_pos_examples)
                    pos_samples.append(_pos_examples[0])
                    remainings += _pos_examples[1:]

                if len(_neg_examples) > 0:
                    random.shuffle(_neg_examples)
                    neg_samples.append(_neg_examples[0])
                    remainings += _neg_examples[1:]

            # at most num//2 pos and neg for remaining
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)
            if num % 2 == 0:
                pos_size, neg_size = num // 2, num // 2
            else:
                pos_size, neg_size = num // 2 + 1, num // 2

            # adjust the sizes
            if len(pos_samples) < pos_size:
                diff = pos_size - len(pos_samples)
                pos_size -= diff
                neg_size += diff

            samples += pos_samples[:pos_size]
            remainings += pos_samples[pos_size:]
            samples += neg_samples[:neg_size]
            remainings += neg_samples[neg_size:]
        case _:
            logging.warning(f"Undefined type: {_type}")

    return samples, remainings


def main(args):
    # load data
    with open(args.filepath_input, "r") as f:
        examples = json.load(f)

    logging.info("#" * 20 + " Stats of all examples " + "#" * 20)
    stats(examples["examples"])

    type_id_examples = defaultdict(lambda: defaultdict(list))
    for example in examples["examples"]:
        metadata = example["metadata"]
        idx = f"{metadata['user_id']}-{metadata['toy_id']}-{metadata['recording_id']}"
        type_id_examples[example["type"]][idx].append(example)

    type2num = get_group_size(args.num_total_sample)
    logging.info(f"{type2num=}")

    random.seed(args.seed)

    all_samples, all_remainings = [], []
    for _type, num in type2num.items():
        samples, remainings = sample(_type, deepcopy(type_id_examples[_type]), num)
        all_samples += samples
        all_remainings += remainings

    logging.info("#" * 20 + " Stats of samples " + "#" * 20)
    stats(all_samples)

    if len(all_samples) + len(all_remainings) != len(examples["examples"]):
        logging.warning(
            f"{len(all_samples)=} + {len(all_remainings)=} != {len(examples['examples'])=}"
        )

    logging.info("Output...")
    output_samples = {
        "metadata": {
            "data-created": datetime.today().strftime("%Y-%m-%d"),
            "total": len(all_samples),
        },
        "examples": all_samples,
    }
    with open(args.dirpath_output / f"samples_{args.num_total_sample}.json", "w") as f:
        json.dump(output_samples, f, indent=4)
        f.write("\n")

    output_remainings = {
        "metadata": {
            "data-created": datetime.today().strftime("%Y-%m-%d"),
            "total": len(all_remainings),
        },
        "examples": all_remainings,
    }
    with open(
        args.dirpath_output / f"remainings_{args.num_total_sample}.json", "w"
    ) as f:
        json.dump(output_remainings, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Sample examples")
    parser.add_argument("--filepath_input", type=Path, help="filepath to input data")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output")
    parser.add_argument("--num_total_sample", type=int, help="num total sample")
    parser.add_argument("--seed", type=int, help="random seed", default=7)
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "sample.log"),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
