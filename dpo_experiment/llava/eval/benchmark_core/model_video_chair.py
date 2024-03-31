import argparse
import logging

import copy
import codecs
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import decord
import os
import json
import random
import requests
from tqdm import tqdm
import numpy as np

from llava.constants import MM_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.utils.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, \
    get_frame_indices
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.model.multimodal_encoder.processor import Blip2ImageTrainProcessor

from transformers import CLIPImageProcessor
from PIL import Image
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")


def get_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


def load_frames(frames_dir, frame_names):
    results = []
    for frame_name in frame_names:
        image_path = f"{frames_dir}/{frame_name}"
        image = get_image(image_path)
        results.append(image)
    return results


def uniform_sample(frames):
    """
    Uniformly samples 10 frames from a list of frames.

    Args:
    - frames (list): A list of frames.

    Returns:
    - list: A list containing 10 uniformly sampled frames.
    """

    total_frames = len(frames)

    # Calculate the sampling interval
    interval = (total_frames - 1) / 9

    # Sample 10 frames
    sampled_frames = [frames[int(i * interval)] for i in range(10)]

    return sampled_frames


def sample_frames(frames, num_segments):
    frame_indices = list(range(len(frames)))
    cand_indices = copy.deepcopy(frame_indices)
    intervals = np.linspace(start=0, stop=len(frame_indices), num=num_segments + 1).astype(int)
    ranges = []

    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    try:
        frame_indices = [cand_indices[random.choice(range(x[0], x[1]))] for x in ranges]
    except:
        frame_indices = [cand_indices[x[0]] for x in ranges]

    sampled_frames = [frames[indice] for indice in frame_indices]

    return sampled_frames


def run_inference(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map={"":0})
    image_processor = Blip2ImageTrainProcessor(
        image_size=model.config.img_size,
        is_training=False)

    # result_dir = os.path.join(args.model_path, 'video_chair')
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    # Load the ground truth file
    # image_files = [os.path.join(args.data_path, filename) for filename in os.listdir(args.data_path)][:args.num_images]

    output_list = []  # List to store the output results
    valid_data = json.load(open(args.validation_data))

    for i, data in tqdm(enumerate(valid_data)):
        if args.num_samples > 0 and i >= args.num_samples:
            print(f"finished, exit.")
            break

        question = "Describe the following video in detail."

        # Question input here
        qs = question
        # qs = DEFAULT_VIDEO_TOKEN + '\n' + qs
        if model.config.mm_use_start_end:
            qs = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VIDEO_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_VIDEO_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # inputs = tokenizer([prompt])
        input_ids = tokenizer_image_token(prompt, tokenizer, MM_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()

        frame_folder = data['obj_info']['vis_path']
        selected_frames = data['obj_info']['frame_name']

        # try:
        images = load_frames(frame_folder, selected_frames)
        if len(images) > args.num_segments:
            images = sample_frames(images, args.num_segments)
        if isinstance(image_processor, CLIPImageProcessor):
            images_tensor = [image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in
                             images]
        else:
            images_tensor = [image_processor.preprocess(image) for image in images]
        images_tensor = torch.stack(images_tensor, dim=0).half().cuda()
        # print(images_tensor.shape)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # images_tensors = [images_tensor.clone() for _ in range(args.num_beams)]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images= [images_tensor],
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                # num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(conv.sep):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        data['masp_inference'] = outputs
        output_list.append(data)
        # except Exception as e:
        #     print(f"Error processing video {video_name}: {e}")
        #     continue

   # with open(os.path.join(result_dir, args.output_file), 'w') as file:
    with codecs.open(os.path.join(result_dir, args.output_file), 'w', encoding='utf-8') as file:
        json.dump(output_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Directory containing video files.', type=str, default="")
    parser.add_argument('--validation_data', type=str,
                        default="/mnt/bn/algo-masp-nas-2/baiyi.by/data/video_chair/obj_and_policy_eval_v0_new.json")
    parser.add_argument('--num_samples', help='Number of samples to predict', type=int, default=-1)
    parser.add_argument("--model_path", type=str,
                        default="/mnt/bn/yukunfeng-nasdrive/xiangchen/model/masp_models/checkpoints/mistral-m3it-ttvqa-7k-correctprompt_2")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--conv_mode", type=str, default="mistral")
    parser.add_argument("--output_file", type=str, default="vid_top1k_res.json")
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    run_inference(args)
