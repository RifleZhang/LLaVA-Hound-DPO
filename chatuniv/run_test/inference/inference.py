import socket
import json
from PIL import Image
import fire
import os
from tqdm import tqdm
import torch 
from logzero import logger
import random
import math
import numpy as np

from run_test.utils import load_jsonl, save_jsonl, load_json, save_json

from transformers import CLIPImageProcessor
from PIL import Image

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_ranged_data(data, range_start, range_end):
    start_idx = int(len(data) * range_start)
    end_idx = int(len(data) * range_end)
    return data[start_idx:end_idx]

def inference_data_list(model_inference, data_list, output_path, proc_func, **kwargs):
    if os.path.exists(output_path):
        res = load_jsonl(output_path)
        res_idx = set([x['id'] for x in res])
        ll = len(data_list)
        logger.info(f"load {len(res)}, full chunck length: {ll}, need process length: {ll - len(res)}")
    else:
        res_idx = set()

    fout = open(output_path, 'a')
    for i, item in tqdm(enumerate(data_list), total=len(data_list)):
        data_to_send = proc_func(item, **kwargs)
        if data_to_send['id'] in res_idx:
            continue
        try:
            resulting_output = model_inference.generate(question=data_to_send['query'], 
                                                        video_path=data_to_send['video_path'],)
            data_to_send['model_prediction'] = {
                'status': 'success',
                'message': resulting_output,
            }
            if i < 100:
                print(resulting_output)
        except Exception as e:
            logger.error(f"error {e} for {data_to_send['id']}")
            # model prediction as error message
            data_to_send['model_prediction'] = {
                'status': 'error',
                'message': str(e),
            }
        fout.write(json.dumps(data_to_send) + '\n')
