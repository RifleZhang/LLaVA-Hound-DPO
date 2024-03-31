import math
import os
import argparse
import json

import numpy as np

import torch
import transformers
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

detail_templates = [
    "Provide a comprehensive analysis of the video's content and themes.",
    "Elaborate on the visual and narrative elements of the video in detail.",
    "Describe every scene and its significance in the video.",
    "Share a detailed breakdown of the video's storyline and visuals.",
    "Explain the video's components, including its characters, setting, and plot.",
    "Offer a detailed interpretation of the video's message and imagery.",
    "Narrate the video's unfolding events in a descriptive manner.",
    "Analyze the video, focusing on its cinematography and narrative structure.",
    "Dissect the video's content, explaining each element thoroughly.",
    "Walk through the video, detailing its key moments and features.",
    "Write an in-depth depiction of the video, covering all its aspects.",
    "Detail the video's plot development, character arcs, and thematic elements.",
    "Illustrate the video's narrative journey, scene by scene, with attention to detail.",
    "Provide an exhaustive description of the video content.",
    "Elaborate on all aspects of the video you are viewing.",
    "Convey the narrative and visual elements of the video in detail.",
    "Explore the thematic and visual aspects of the video comprehensively.",
    "Write a comprehensive depiction of the entire video clip.",
    "Characterize each scene of the video using a detailed description."
]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--legacy", type=bool, default=False)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args, temperature=0, top_p=0.9, max_new_tokens=512):
    if model.config.mm_use_x_start_end:
        qs = DEFAULT_X_START_TOKEN['VIDEO'] + DEFAULT_X_TOKEN['VIDEO'] + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + qs
    else:
        qs = DEFAULT_X_TOKEN['VIDEO'] + '\n' + qs

    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    '''
    images (X_modalities) [
            [img_feature, img_feature, video_feature, audio_feature],
            ['image', 'image', 'video', 'audio']
            ]
    '''

    if temperature < 0.01:
        temperature = -1 # greedy
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(max_context_length - input_ids.shape[1], max_new_tokens)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[[video_tensor], ['video']],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
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
    print(outputs)
    return outputs

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, legacy=args.legacy)
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        id = sample['question_id']
        # random sample for detail template
        question = np.random.choice(detail_templates)

        sample_set = {'id': id, 'question': question}

        temp_path = os.path.join(args.video_dir, video_name)
        if not os.path.exists(temp_path):
            continue
        video_path = temp_path

        # # generate detailed descriptions
        # n_detail_templates = len(detail_templates)
        # rand_idx = np.random.randint(n_detail_templates)
        # detail_question = detail_templates[rand_idx]

        # Run inference on the video and add the output to the list
        output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
        # output = get_model_output(model, processor['video'], tokenizer, video_path, detail_question, args)
        sample_set['pred'] = output
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()
    # Save the output list to a JSON file


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
