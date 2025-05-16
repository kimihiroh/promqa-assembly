import os

from google import genai

from src.utils.api import (
    encode_image,
)

from utils import sample_frame


def format_content_openai(args, example, template_components, toy2instruction):
    content = []

    # prefix
    content.append({"type": "text", "text": template_components["prefix"]})

    # parts
    content.append({"type": "text", "text": template_components["parts"]})
    filepath_parts = toy2instruction[example['toy_id']]['parts']
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(filepath_parts)}",
            },
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
        args.max_frames,
    )
    for filepath in sampled_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(example['frames'][filepath])}",
                },
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


def format_content_anthropic(args, example, template_components, toy2instruction):
    content = []

    # prefix
    content.append({"type": "text", "text": template_components["prefix"]})

    # parts
    content.append({"type": "text", "text": template_components["parts"]})
    filepath_parts = toy2instruction[example['toy_id']]['parts']
    content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encode_image(filepath_parts),
            },
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
        args.max_frames,
    )
    for filepath in sampled_frames:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encode_image(example['frames'][filepath]),
                },
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


def format_content_google(args, example, template_components, toy2instruction):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    content = []
    text_prompt = ""
    uploaded_files = []

    # prefix
    content.append(template_components["prefix"])
    text_prompt += template_components["prefix"] + "\n"

    # parts
    content.append(template_components["parts"])
    text_prompt += template_components["parts"] + "\n"
    filepath_parts = toy2instruction[example['toy_id']]['parts']
    ref_filepath_parts = client.files.upload(file=filepath_parts)
    content.append(ref_filepath_parts)
    uploaded_files.append(ref_filepath_parts)

    # assembly manual
    manual = template_components["manual"]["dot"]
    manual = manual.replace("{dot}", toy2instruction[example["toy_id"]]["dot"])
    content.append(manual)
    text_prompt += manual + "\n"

    # recording
    content.append(template_components["recording"])
    text_prompt += template_components["recording"] + "\n"
    sampled_frames = sample_frame(
        example["video"]["start"],
        example["video"]["end"],
        args.max_frames,
    )
    for filepath in sampled_frames:
        ref_filepath = client.files.upload(file=example['frames'][filepath])
        content.append(ref_filepath)
        uploaded_files.append(ref_filepath)

    # task
    content.append(template_components["task"].replace("{question}", example["question"]))
    text_prompt += (
        template_components["task"].replace("{question}", example["question"]) + "\n"
    )

    return content, text_prompt.strip(), uploaded_files
