"""
API-related functions

"""

import base64
from collections import defaultdict
import logging
import os
from pathlib import Path
import time

import anthropic
from google import genai
from google.genai import types
from openai import OpenAI

PRICE = {
    "gpt-4o-2024-08-06": {
        "input": 2.5 / 1e6,
        "output": 10 / 1e6,
    },
    "gpt-4o-2024-11-20": {
        "input": 2.5 / 1e6,
        "output": 10 / 1e6,
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15 / 1e6,
        "output": 0.6 / 1e6,
    },
    "o1-2024-12-17": {
        "input": 15 / 1e6,
        "output": 60 / 1e6,
    },
    "o3-mini-2025-01-31": {
        "input": 1.1 / 1e06,
        "output": 4.4 / 1e06,
    },
    "gpt-4.5-preview-2025-02-27": {
        "input": 75 / 1e6,
        "output": 150 / 1e6,
    },
    "claude-3-5-sonnet-20240620": {
        "input": 3 / 1e6,
        "output": 15 / 1e6,
    },
    "claude-3-7-sonnet-20250219": {
        "input": 3 / 1e6,
        "output": 15 / 1e6,
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.8 / 1e6,
        "output": 4 / 1e6,
    },
    "gemini/gemini-1.5-pro-001": {
        # the price doubles for >128k tokens
        "input": 3.5 / 1e6,
        "output": 10.5 / 1e6,
    },
    "gemini-2.5-pro-exp-03-25": {
        "input": 0 / 1e6,
        "output": 0 / 1e6,
    },
    "gemini-2.5-pro-preview-05-06": {
        "input": 1.25 / 1e6,
        "output": 10 / 1e6,
    },
    "gemini-2.0-flash-001": {
        "input": 0.1 / 1e6,
        "output": 0.4 / 1e6,
    },
}


def call_openai_api_single(
    model_id: str,
    messages: list,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    call OpenAI API, given single "messages"

    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    output = ""
    count_tokens = defaultdict(int)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output = response.choices[0].message.content
        count_tokens["input"] += response.usage.prompt_tokens
        count_tokens["output"] += response.usage.completion_tokens
    except Exception as e:
        output = f"Exception: {e}"
        logging.warning(f"Exception: {e}")

    return output, count_tokens


def call_anthropic_api_single(
    model_id: str,
    messages: list,
    temperature: float,
    max_tokens: int,
):
    client = anthropic.Anthropic()

    output = ""
    count_tokens = defaultdict(int)
    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        output = response.content[0].text
        count_tokens["input"] += response.usage.input_tokens
        count_tokens["output"] += response.usage.output_tokens
    except Exception as e:
        output = f"Exception: {e}"
        logging.warning(f"Exception: {e}")

    return output, count_tokens


def call_google_api_single(
    model_id: str, content: list, temperature: float, max_tokens: int, files: list
):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    output, count_tokens = "", defaultdict(int)
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=content,
            config=types.GenerateContentConfig(
                # max_output_tokens=max_tokens,  # no output if enabled, bug?
                temperature=temperature,
            ),
        )
        output = response.text
        count_tokens["input"] += response.usage_metadata.prompt_token_count
        count_tokens["output"] += response.usage_metadata.candidates_token_count
    except Exception as e:
        output = f"Exception: {e}"
        logging.warning(f"Exception: {e}")

    logging.info("Delete uploaded images ...")
    for file in files:
        client.files.delete(name=file.name)

    return output, count_tokens


def call_openai_api(
    model_id: str,
    messages_list: list[str],
    temperature: float,
    max_tokens: int,
    wait_time: int = 30,
) -> list[str]:
    """call OpenAI API"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    responses = []
    count_tokens = defaultdict(int)
    for messages in messages_list:
        try:
            if model_id == "o1-2024-12-17":
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[messages],
                    reasoning_effort="medium",  # hard-coded, default
                )
                # temperature, max token cannot be set
            elif model_id == "o3-mini-2025-01-31":
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[messages],
                )
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            responses.append(response.choices[0].message.content)
            count_tokens["input"] += response.usage.prompt_tokens
            count_tokens["output"] += response.usage.completion_tokens
        except Exception as e:
            responses.append((model_id, "Error"))
            logging.info(f"Exception: {e}")
        # todo: change here, more dynamically adjust wait time
        time.sleep(wait_time)

    estimate_cost(model_id, count_tokens)

    return responses


def call_anthropic_api(
    model_id: str,
    messages_list: list[str],
    temperature: float,
    max_tokens: int,
    wait_time: int = 30,
):
    client = anthropic.Anthropic()

    responses = []
    count_tokens = defaultdict(int)
    for messages in messages_list:
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[messages],
            )
            responses.append(response.content[0].text)
            count_tokens["input"] += response.usage.input_tokens
            count_tokens["output"] += response.usage.output_tokens
        except Exception as e:
            responses.append((model_id, "Error"))
            logging.info(f"Exception: {e}")
        # todo: change here, more dynamically adjust wait time
        time.sleep(wait_time)

    estimate_cost(model_id, count_tokens)

    return responses


def call_google_api(
    model_id: str,
    messages_list: list[str],
    temperature: float,
    max_tokens: int,
    wait_time: int = 30,
):
    """
    Note: image input is not implemented now
    """

    responses = []
    count_tokens = defaultdict(int)
    for messages in messages_list:
        try:
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            assert len(messages["content"]) == 1
            response = client.models.generate_content(
                model=model_id,
                contents=[messages["content"][0]["text"]],
                config=types.GenerateContentConfig(
                    # max_output_tokens=max_tokens,  # no output if enabled, bug?
                    temperature=temperature,
                ),
            )
            responses.append(response.text)
            count_tokens["input"] += response.usage_metadata.prompt_token_count
            count_tokens["output"] += response.usage_metadata.candidates_token_count
        except Exception as e:
            responses.append((model_id, "Error"))
            logging.info(f"Exception: {e}")
        # todo: change here, more dynamically adjust wait time
        time.sleep(wait_time)

    estimate_cost(model_id, count_tokens)

    return responses


def estimate_cost(model_id: str, count: dict[str, int]) -> float:
    """estimate cost"""
    cost = (
        PRICE[model_id]["input"] * count["input"]
        + PRICE[model_id]["output"] * count["output"]
    )
    logging.info(f"Estimated cost: ${cost:.4f}.")
    return cost


def encode_image(filepath: Path):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
