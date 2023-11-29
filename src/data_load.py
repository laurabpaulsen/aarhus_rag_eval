#!/usr/bin/env python3

import json
from pathlib import Path

default_path = Path(__file__).parents[1] / "data" / "loop_q_and_a_w_meta.json"

def load_loop(path: Path = default_path) -> list[dict]:
    with open(path) as f:
        jsondata = json.load(f)
    return jsondata


def map_filter(jsondata: list[dict], field: str) -> list[str]:
    return [item[field] for item in jsondata]

def map_questions(jsondata: list[dict]) -> list[str]:
    return map_filter(jsondata, 'question')


if __name__ == '__main__':
    jsondata = load_loop()
    assert len(jsondata) == 700
