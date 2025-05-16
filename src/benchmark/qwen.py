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

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch

from utils import (
    format_instruction,
    sample_frame,
)

from src.utils.misc import get_date


def load_model(args):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    # load model
    model = LLM(
        model=args.model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"image": 1, "video": 1},
        dtype=torch.bfloat16,
        seed=args.seed,
        max_model_len=32768,
    )

    # load processor
    processor = AutoProcessor.from_pretrained(args.model_id)

    return model, processor, sampling_params


def format_content_qwen(args, example, template_components, toy2instruction):
    content = []

    # prefix
    content.append({"type": "text", "text": template_components["prefix"]})

    # parts
    content.append({"type": "text", "text": template_components["parts"]})
    filepath_parts = toy2instruction[example['toy_id']]['parts']
    content.append(
        {
            "type": "image",
            "image": str(filepath_parts),
        }
    )

    # assembly manual
    content.append(
        {
            "type": "text",
            "text": template_components["manual"]["dot"].replace(
                "{dot}", toy2instruction[example["toy_id"]]["dot"]
            ),
        }
    )

    # recording
    content.append({"type": "text", "text": template_components["recording"]})
    sampled_frames = sample_frame(
        example["video"]["start"],
        example["video"]["end"],
        args.max_frame,
    )
    content.append(
        {
            "type": "video",
            "video": [str(filepath) for filepath in sampled_frames],
        }
    )

    # task
    content.append(
        {
            "type": "text",
            "text": template_components["task"].replace(
                "{question}", example["question"]
            ),
        }
    )

    text_prompt = ""
    for _content in content:
        if _content["type"] == "text":
            text_prompt += _content["text"] + "\n"
        else:
            text_prompt += _content["type"] + "\n"

    return content, text_prompt.strip()


def format_input(
    args,
    examples,
    template_components,
    toy2instruction,
):
    logging.info("Prepare input ... ")
    messages_list = []
    for idx, example in tqdm(enumerate(examples), total=len(examples)):
        content, text_prompt = format_content_qwen(
            args, example, template_components, toy2instruction
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        messages_list.append(messages)

    return messages_list


def predict(model, processor, sampling_params, messages_list):
    llm_inputs = []
    for messages in messages_list:
        prompt_processed = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # image_inputs, video_inputs, video_kwargs = process_vision_info(
        #     messages, return_video_kwargs=True
        # )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        # llm_inputs.append(
        #     {
        #         "prompt": prompt_processed,
        #         "multi_modal_data": mm_data,
        #         # FPS will be returned in video_kwargs
        #         "mm_processor_kwargs": video_kwargs,
        #     }
        # )
        llm_inputs.append(
            {
                "prompt": prompt_processed,
                "multi_modal_data": mm_data,
            }
        )

    logging.info("Prediction starts ... ")
    size = 40
    outputs = []
    for i in range(0, len(llm_inputs), size):
        _outputs = model.generate(llm_inputs[i : i + size], sampling_params)
        outputs += [x.outputs[0].text for x in _outputs]

    return outputs


def save(args, examples, messages_list, predictions):
    new_examples = []
    for example, messages, prediction in zip(examples, messages_list, predictions):
        new_example = deepcopy(example)
        new_example["prediction"] = {
            "text_prompt": messages[-1]["content"],
            "response": prediction,
        }
        new_examples.append(new_example)

    filepath_output = (
        args.dirpath_output
        / f"open_multimodal_{Path(args.model_id).name}_{args.resolution}_{args.max_frame}_{args.color}_{args.angle}.json"
    )

    output = {
        "metadata": {
            "data-created": get_date(),
            "model_id": args.model_id,
            "max_frame": args.max_frame,
            "input": args.filepath_input.name,
        },
        "examples": new_examples,
    }

    with open(filepath_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


def main(args):
    
    examples = []
    for toy_id in SUBSETS:
        examples += load_dataset("kimihiroh/promqa-assembly", toy_id, split='test')

    toy2instruction = format_instruction()

    with open(args.filepath_template, "r") as f:
        template_components = yaml.safe_load(f)

    messages_list = format_input(
        args,
        examples,
        template_components,
        toy2instruction,
    )

    logging.info("Sanity Check ...")
    idx = 1
    logging.info(f"{idx=}")
    for _content in messages_list[idx][-1]["content"]:
        logging.info(_content[_content["type"]])

    model, processor, sampling_params = load_model(args)

    predictions = predict(
        model,
        processor,
        sampling_params,
        messages_list,
    )

    save(args, examples, messages_list, predictions)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code for Qwen 2.5 VL")
    
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
    parser.add_argument("--max_frame", type=int, help="max #frame", default=64)
    parser.add_argument("--angle", type=str, help="angle", default="C10115_rgb")
    parser.add_argument("--resolution", type=str, help="resolution", default="360p")
    parser.add_argument("--color", type=str, help="color", default="rgb")
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
                args.dirpath_log / f"open_multimodal_qwen25vl_{get_date()}.log"
            ),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")

    main(args)
