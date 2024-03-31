import argparse
import logging
from pickle import FALSE
from accelerate import Accelerator
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import decord
import os
import json
import random
import requests
from tqdm import tqdm
import numpy as np

from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# def save_jsonl(path, line):
#     with open(path, 'a') as f:
#         f.write(json.dumps(line) + '\n')

class EvalDataset(Dataset):
    def __init__(self, args, tokenizer, video_processor, device="cude") -> None:
        super().__init__()
        self.args = args
        self.valid_data = json.load(open(args.validation_data))
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.device = device
    
    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, index):
        # Question input here
        inp = "Describe the following video in detail."
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        print(f"{roles[1]}: {inp}")
        inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_X_token(prompt, self.tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()

        frame_folder = self.valid_data[index]['obj_info']['vis_path']
        video_tensor = self.video_processor(frame_folder, return_tensors='pt', video_decode_backend='frames')['pixel_values']
        if type(video_tensor) is list:
            images_tensor = [video.to(self.device, dtype=torch.float16) for video in video_tensor]
        else:
            images_tensor = video_tensor.to(self.device, dtype=torch.float16)

        return input_ids, images_tensor, torch.tensor(index)
        
    def save_output(self, res_lst, result_dir):
        output_lst = []
        idx_lst = []
        for idx_batch, pred_batch in res_lst:
            batch_size = idx_batch.size(0)
            # print(idx_batch, batch_size)
            for i in range(batch_size):
                idx = int(idx_batch[i])
                if idx in idx_lst:
                    continue
                idx_lst.append(idx)
                pred = pred_batch[i]
                case_info = self.valid_data[idx].copy()
                case_info.update({'masp_inference' : pred})
                output_lst.append(case_info)
        with open(os.path.join(result_dir, self.args.output_file), 'w', encoding='utf-8') as file:
            json.dump(output_lst, file, ensure_ascii=False, indent=4)

def run_inference(args):
    # disable_torch_init()
    accelerator = Accelerator()
    # rank = accelerator.rank
    # print(f"local rank {rank}")
    model_path = args.model_path
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    video_processor = processor['video']

    eval_dataset = EvalDataset(args, tokenizer, video_processor, device=model.device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    model.eval()
    # Load the ground truth file
    # image_files = [os.path.join(args.data_path, filename) for filename in os.listdir(args.data_path)][:args.num_images]
    key = ['video']
    res = []  # List to store the output results
    for input_ids, images_tensor, data_idx in tqdm(eval_dataloader):
        input_ids = input_ids[0].to(accelerator.device)
        images_tensor = images_tensor[0].to(accelerator.device)
        data_idx = data_idx[0].to(accelerator.device)
        # accelerator.print(data_idx)
        # input_ids = input_ids[0].cuda()
        # images_tensor = images_tensor[0].cuda()

        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        output_ids = model.generate(
            input_ids,
            images=[images_tensor, key],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
        # output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        output_ids_padded = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.eos_token_id)
        output_ids_gathered = accelerator.gather(output_ids_padded)
        input_len_gathered = accelerator.gather(torch.tensor([input_ids.shape[1]], device=accelerator.device))
        data_idx_gathered = accelerator.gather(data_idx)
        outputs = []
        for i in range(input_len_gathered.shape[0]):
            input_token_len = input_len_gathered[i].cpu()
            output = tokenizer.batch_decode(output_ids_gathered[i:i+1, input_token_len:], skip_special_tokens=True)[0]
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()            
            outputs.append(output)

        res.append((data_idx_gathered.cpu(), outputs))
        # accelerator.print(f'sync************ {data_idx.cpu().item()}')
    
    # accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        eval_dataset.save_output(res, result_dir)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Directory containing video files.', type=str, default="")
    parser.add_argument('--validation_data', type=str,
                        default="/mnt/bn/algo-masp-nas-2/baiyi.by/data/video_chair/obj_and_policy_eval_v0_new.json")
    parser.add_argument('--num_images', help='Number of images to predict', type=int, default=100)
    parser.add_argument("--model_path", type=str,
                        default="/mnt/bn/yukunfeng-nasdrive/xiangchen/model/masp_models/checkpoints/llava-mistral-videoproj-pretrain-frames-base-intvid-ft-m3it-img-ttvqa-7k")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--output_file", type=str, default="vid_top1k_res.json")
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    args = parser.parse_args()
    run_inference(args)
