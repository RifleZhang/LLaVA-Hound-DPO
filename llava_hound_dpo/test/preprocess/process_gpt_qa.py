import os, sys, time, os.path as osp
import math
import numpy as np
import glob
import json
from collections import defaultdict 
from tqdm import tqdm

import pandas as pd

import pathlib
from typing import Any, Dict, List, Optional, Union
import fire
from logzero import logger
from data_processing.utils import load_json_data, save_jsonl, save_json, load_pickle
from data_processing.utils import parse_single_file

import re

def make_conv_video(prompt, answer):
    prompt = "<video>\n" + prompt.strip()
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

def parser(text):
   
    # Pattern to match question and answer pairs, robust to extra whitespace or newlines after "Q" or "A"
    pattern = r'Q(\d+)\s*:\s*(.*?)\n\s*A\1\s*:\s*(.*?)\n*(?=\nQ|\Z)'

    # Using re.DOTALL to make '.' match any character including newline
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 3:
        return None
    
    qas = []
    for i, q, a in matches:
        qas.append(make_conv_video(q, a))
    
    return qas



def main(data_path, output_path, frame_data_path, **kwargs):
    data = load_json_data(data_path)
    frame_data = load_json_data(frame_data_path)
    frame_dict = {item['id']: item['video'] for item in frame_data}

    qa_data = []
    for item in tqdm(data):
        idx = item['id']
        if idx not in frame_dict:
            print(f"no frame data for {idx}")
            continue
        path = frame_dict[idx]
        response = item['response']
        try:
            parsed_result = parser(response)
        except:
            print(f"invalid item: {item}")
            continue
        if parsed_result is None:
            continue
        for i, qa in enumerate(parsed_result):
            qa_data.append(
                {
                    "id": f"{idx}_{i}",
                    "video": path,
                    "conversations": qa,
                }
            )
        # for qa in parsed_result:
        #     qa_data.append(
        #         {
        #             "idx": idx,
        #             "video": path,
        #             "conversations": qa,
        #         }
        #     )
    save_jsonl(output_path, qa_data)

    # model_caption_data = []
    # for item in data:
    #     model_caption_data.append(parse_single_file(item))


if __name__ == "__main__":
    fire.Fire(main)