"""
miscellaneous functions

"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Optional
import logging


def get_date(granularity: Optional[str] = "min") -> str:
    """
    get date

    """
    date_time = datetime.now()
    if granularity == "min":
        str_data_time = date_time.strftime("%Y%m%d-%H%M")
    elif granularity == "day":
        str_data_time = date_time.strftime("%Y%m%d")
    else:
        logging.error(f"Undefined timestamp granularity: {granularity}")

    return str_data_time


def to_dict(item: Any) -> Any:
    if isinstance(item, defaultdict):
        if None in item:
            return {k: to_dict(v) for k, v in item.items()}
        else:
            return {k: to_dict(v) for k, v in sorted(item.items())}
    return item


def format_time(seconds: float, time_format: str = "all") -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if time_format == "all":
        output = f"{int(h):02}:{int(m):02}:{s:06.3f}"
    elif time_format == "short":
        output = f"{int(m):02}:{int(s):02}"
    else:
        logging.error(f"Undefined time_format: {time_format}")
        output = None

    return output
