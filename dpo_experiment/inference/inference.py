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

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

from data_processing.utils import load_jsonl, save_jsonl, load_json, save_json

from transformers import CLIPImageProcessor
from PIL import Image

def get_id_from_path(path):
    return path.split('/')[-1].split('.')[0]

MODAL_TOKEN_LIST=["<video>", "<image>"
                  ]
def remove_special_tokens(text):
    for token in MODAL_TOKEN_LIST:
        if token in text:
            text = text.replace(token, "").strip()
    return text

def model_function(model_dict, input_data):
    """
        input_data:
        {
            "modal_type": "VIDEO" or "IMAGE"
            "query": "query",
            "modal_path": image or video path,
            "video_decode_backend": "frames" or "decord" # for video
        }
    """
    # unpack model dict
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    processor = model_dict["processor"]
    video_processor = processor.get('video', None)
    image_processor = processor.get('image', None)
    context_len = model_dict["context_len"]
    modal_type = input_data.get('modal_type', 'VIDEO').upper()

    qs = remove_special_tokens(input_data['query'])
    if model.config.mm_use_x_start_end:
        qs = DEFAULT_X_START_TOKEN[modal_type] + DEFAULT_X_TOKEN[modal_type] + DEFAULT_X_END_TOKEN[modal_type] + '\n' + qs
    else:
        qs = DEFAULT_X_TOKEN[modal_type] + '\n' + qs
    # print(qs)

    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if modal_type == 'IMAGE':
        image_path = input_data['modal_path']
        modal_tensor = image_processor.preprocess(image_path, return_tensors='pt')['pixel_values'][0].half().to('cuda')
    elif modal_type == 'VIDEO':
        video_decode_backend = input_data.get('video_decode_backend', 'frames')
        video_path = input_data['modal_path']
        modal_tensor = video_processor(video_path, return_tensors='pt', video_decode_backend=video_decode_backend)['pixel_values'][0].half().to('cuda')
    else:
        raise ValueError(f"modal_type {modal_type} not supported")

    # print(video_tensor.shape)
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[modal_type], return_tensors='pt').unsqueeze(0).to('cuda')

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = input_data.get('temperature', 0.7)
    if temperature < 0.01:
        temperature = -1 # greedy
    top_p = input_data.get('top_p', 0.9)
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = input_data.get("max_new_tokens", 1024)
    max_new_tokens = min(max_context_length - input_ids.shape[1], max_new_tokens)
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[[modal_tensor], [modal_type.lower()]],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    # result = {
    #     'status': 'success',
    #     'message': outputs
    # }
    # # print(outputs)
    # return result

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

def inference_data_list(model_dict, data_list, output_path, proc_func, **kwargs):
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
            resulting_output = model_function(model_dict, data_to_send)
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
    fout.close()