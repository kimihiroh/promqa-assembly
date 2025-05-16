"""
Predict answers from each set of a video (images), a recipe, and a question.

"""

from argparse import ArgumentParser
from copy import deepcopy
import json
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageOps
import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    # DEFAULT_IMAGE_TOKEN,
    # DEFAULT_IM_START_TOKEN,
    # DEFAULT_IM_END_TOKEN,
    # IGNORE_INDEX,
)

from utils import (
    format_instruction,
    sample_frame,
)

from src.utils.misc import get_date


def format_prompt(
    args, example, template_components, toy2instruction, num_frame, timestamps, duration
):
    # prefix
    prompt = template_components["prefix"] + "\n"

    # assembly manual
    prompt += (
        template_components["manual"]["dot"].replace(
            "{dot}", toy2instruction[example["toy_id"]]["dot"]
        )
        + "\n"
    )

    # token for video
    prompt += "<image>\n"

    # parts & recording
    prompt += f"The video lasts for {duration:.2f} seconds. "
    prompt += "The video starts with an image containing the parts, final picture, and/or exploded view. "
    prompt += (
        f"One frame is sampled corresponding the parts information part, "
        f"then {num_frame} frames are uniformly sampled from the the user's activity recording part in the video. "
        f"These frames are located at {timestamps}."
    ) + "\n"

    # task
    prompt += template_components["task"].replace("{question}", example["question"])

    prompt_in_template = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return prompt_in_template


def load_frames(args, example):
    sampled_frame_filepaths = sample_frame(
        example["video"]["start"],
        example["video"]["end"],
        args.max_frames,
    )

    frames = np.array([imageio.imread(filepath) for filepath in sampled_frame_filepaths])

    if sampled_frame_filepaths[0].stem == "0":
        timestamps = ",".join(
            [f"{float(filepath)+1:.2f}s" for filepath in sampled_frame_filepaths]
        )
        duration = example["video"]["end"] + 1 - example["video"]["start"]
    else:
        timestamps = ",".join(
            [f"{float(filepath):.2f}s" for filepath in sampled_frame_filepaths]
        )
        duration = example["video"]["end"] - example["video"]["start"]

    return frames, timestamps, duration


def load_image(filepath, width, height):
    # Open the image
    img = Image.open(filepath)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Add padding
    padded_img = ImageOps.pad(img, (width, height), color=0)

    return np.array(padded_img)


def process_image_and_video(args, example, image_processor, toy2instruction):
    # load video frames
    frames, timestamps, duration = load_frames(args, example)
    num_frame, height, width, _ = frames.shape

    # load parts image
    parts_image = load_image(
        toy2instruction[example['toy_id']]['parts'],
        width=width,
        height=height,
    )

    _parts_image = parts_image[np.newaxis, ...]
    image_video = np.concatenate([_parts_image, frames], axis=0)

    image_video = (
        image_processor.preprocess(image_video, return_tensors="pt")["pixel_values"]
        .cuda()
        .bfloat16()
    )
    image_video = [image_video]

    return image_video, timestamps, duration, num_frame


def format_input_and_predict(
    args,
    examples,
    template_components,
    toy2instruction,
    tokenizer,
    model,
    image_processor,
    device,
):
    metadata = {
        "data-created": get_date(),
        "input": args.filepath_input.name,
        "model_id": args.model_id,
        "max_frames": args.max_frames,
        "angle": args.angle,
    }

    filepath_output = (
        args.dirpath_output
        / f"{Path(args.model_id).name}_{args.resolution}_{args.max_frames}_{args.color}_{args.angle}.json"
    )
    if filepath_output.exists():
        with open(filepath_output, "r") as f:
            examples_prev = json.load(f)
    else:
        examples_prev = []

    new_examples = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        if examples_prev and idx < len(examples_prev["examples"]):
            new_examples.append(examples_prev["examples"][idx])
            continue

        # process image and video
        image_video, timestamps, duration, num_frame = process_image_and_video(
            args,
            example,
            image_processor,
        )

        # fomrat text prompt
        text_prompt = format_prompt(
            args,
            example,
            template_components,
            toy2instruction,
            num_frame + 1,
            timestamps,
            duration,
        )

        # tokenize
        input_ids = (
            tokenizer_image_token(
                text_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device)
        )

        # predict
        cont = model.generate(
            input_ids,
            images=image_video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=args.max_tokens,
        )
        response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        # save
        new_example = deepcopy(example)
        new_example["prediction"] = {
            "text_prompt": text_prompt,
            "response": response,
        }
        new_examples.append(new_example)

        output = {"metadata": metadata, "examples": new_examples}
        with open(filepath_output, "w") as f:
            json.dump(output, f, indent=4)
            f.write("\n")

    return None


def main(args):
    warnings.filterwarnings("ignore")

    examples = []
    for toy_id in SUBSETS:
        examples += load_dataset("kimihiroh/promqa-assembly", toy_id, split='test')

    toy2instruction = format_instruction()

    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_id,
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation=None,
    )
    model.eval()
    # tokenizer, model, image_processor, max_length = None, None, None, None

    format_input_and_predict(
        args,
        examples,
        template_components,
        toy2instruction,
        tokenizer,
        model,
        image_processor,
        args.device,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code for LLaVA-Video")
    
    parser.add_argument("--model_id", type=str, help="model id vision")
    
    # parser.add_argument("--filepath_input", type=Path, help="filepath for input")
    # parser.add_argument(
    #     "--filepath_instruction", type=Path, help="filepath for instruction"
    # )
    # parser.add_argument(
    #     "--dirpath_instruction_image", type=Path, help="dirpath for instruction (image)"
    # )
    # parser.add_argument(
    #     "--dirpath_parts_image", type=Path, help="dirpath for parts (image)"
    # )
    # parser.add_argument("--dirpath_frame", type=Path, help="dirpath for frames")
    parser.add_argument(
        "--filepath_template", type=Path, help="filepath for prompt template",
        default='.src/benchmark/template.yaml'
    )
    parser.add_argument(
        "--dirpath_output", type=Path, help="filepath for output",
        default='./output/prediction'
    )
    
    parser.add_argument("--temperature", type=float, help="temperature", default=0.0)
    parser.add_argument("--top_p", type=float, help="top p", default=1)
    parser.add_argument(
        "--max_tokens", type=int, help="max tokens to generate", default=256
    )
    parser.add_argument("--max_frames", type=int, help="max #frame", default=64)
    parser.add_argument("--angle", type=str, help="angle", default="C10118_rgb")
    parser.add_argument("--resolution", type=str, help="resolution", default="360p")
    parser.add_argument("--color", type=str, help="color", default="rgb")
    parser.add_argument("--device", type=str, help="device: cuda", default="cuda")
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                args.dirpath_log / f"open_multimodal_llava_video_{get_date()}.log"
            ),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
