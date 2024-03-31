from operator import truediv
import os
import re
import json
import sys
import argparse
# from nltk.stem import *
# import nltk
import openai
from abc import ABC, abstractmethod
# from pattern3.en import singularize
# from call_dino_service import 
from tqdm import tqdm

import time

class BaseAPIWrapper(ABC):
    @abstractmethod
    def get_completion(self, user_prompt, system_prompt=None):
        pass

class OpenAIAPIWrapper(BaseAPIWrapper):
    def __init__(self, caller_name="default", key_pool=None, temperature=0, model="gpt-4-32k-0613", time_out=30):
        self.key_pool = key_pool
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        openai.api_base = "https://search-us.byteintl.net/gpt/openapi/online/v2/crawl"
        # openai.api_base = "https://search-us.bytedance.net/gpt/openapi/online/v2/crawl"
        openai.api_type = "azure"
        openai.api_version = "2023-06-01-preview" 
        openai.api_key = key_pool[0]

    def request(self, system_content, usr_question, previous_msg=None, last_answer=None):
        if previous_msg is None:
            msgs = [
                {"role": "system", "content": f"{system_content}"},
                {"role": "user", "content": f"{usr_question}"}
            ]
        else:
            msgs = previous_msg
            msgs += [
                {"role": "assistant", "content": last_answer},
                {"role": "user", "content": usr_question}
            ]
        response = openai.ChatCompletion.create(
            engine="gpt_openapi",
            messages=msgs,
            temperature=self.temperature, 
            model=self.model
        )
        resp = response.choices[0]['message']['content']
        total_tokens = response.usage['total_tokens']

        return resp, msgs, total_tokens
    
    def get_completion(self, user_prompt=None, system_prompt=None, previous_msgs=None, last_answer=None, max_try=20):
        gpt_cv_nlp = '[]'
        key_i = 0
        total_tokens = 0
        # gpt_cv_nlp, total_tokens = self.request(system_prompt, user_prompt)
        while max_try > 0:
            try:
                gpt_cv_nlp, msgs, total_tokens = self.request(system_prompt, user_prompt, previous_msgs, last_answer)
                # print('Succ: ', gpt_cv_nlp)
                max_try = 0
                break
            except Exception as e:
                print(e)
                print("fail ", max_try)
                # key = self.key_pool[key_i%2]
                # openai.api_key = key
                # key_i += 1
                time.sleep(1)
                max_try -= 1
    
        return gpt_cv_nlp, msgs, total_tokens


class CHAIR(object):

    def __init__(self) -> None:
        super().__init__()
        self.system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        self.openai_obj = OpenAIAPIWrapper(key_pool=["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"])
        with open('llava/eval/benchmark_core/prompts/cap2info.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt = content
        with open('llava/eval/benchmark_core/prompts/refine_json.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt_deduplicate = content     
    
    def cap2info_gpt4(self, cap):
        user_prompt = self.cap_user_prompt.replace('/video caption/', cap)
        gpt_ret1, msgs, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)
        user_prompt = self.cap_user_prompt_deduplicate.replace('/json file/', gpt_ret1)
        gpt_ret2, msgs, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt, previous_msgs=msgs, last_answer=gpt_ret1)
        match = re.search(r"(?<=```json\n)([\s\S]*?)(?=```)", gpt_ret2)
        if match:
            info = json.loads(match.group(1))
            # Split the string into a list of items
            return info
        else:
            try:
                start = gpt_ret2.find('{')
                end = gpt_ret2.rfind('}')
                info = json.loads(gpt_ret2[start:end+1])
                return info
            except Exception as e:
                print(gpt_ret1)
                print(gpt_ret2)
                return None

def post_process_masp_cap_label(evaluator, annotations_file, gt=True):
    results = []
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    for data in tqdm(annotations):
        if gt:
            caption = data['refine_caption']
        else:
            caption = data['masp_inference']
        cap_info = evaluator.cap2info_gpt4(caption)
        data['cap_info'] = cap_info
        results.append(data)
    return results


from multiprocessing import Pool



# Function to process a single data item
def process_data(data, evaluator, gt):
    if gt:
        caption = data['refine_caption']
    else:
        caption = data['masp_inference']
    cap_info = evaluator.cap2info_gpt4(caption)
    data['cap_info'] = cap_info
    return data

# Function to initialize the multiprocessing pool and process the data
def process_annotations(evaluator, annotations_file, gt=False, num_sample=None):
    # Load annotations
    annotations = json.load(open(annotations_file))
    if num_sample is not None:
        annotations = annotations[:num_sample]

    # Create a pool of workers equal to the number of available CPU cores
    pool = Pool(processes=8)  # None means use all available cores

    # Use a partial function to fix the gt and evaluator arguments
    from functools import partial
    process_data_partial = partial(process_data, evaluator=evaluator, gt=gt)

    # Map the data processing function over the annotations using the pool
    # pool.map(process_data_partial, annotations)
    res = []
    # for data in tqdm(annotations):
    #     res.append(process_data_partial(data))
    for data in tqdm(pool.imap_unordered(process_data_partial, annotations), total=len(annotations)):
        res.append(data)
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default='/mnt/bn/yukunfeng-nasdrive/xiangchen/model/masp_models/checkpoints/llava-mistral-videoproj-pretrain-frames-base-intvid-ft-m3it-img-ttvqa-7k/video_chair/vid_top1k_res.json')
    parser.add_argument("--output_file", type=str, default='/mnt/bn/yukunfeng-nasdrive/xiangchen/model/masp_models/checkpoints/llava-mistral-videoproj-pretrain-frames-base-intvid-ft-m3it-img-ttvqa-7k/video_chair/vid_top1k_res_info.json')
    parser.add_argument("--gt", type=bool, default=False)
    parser.add_argument("--num_sample", type=int, default=None)

    args = parser.parse_args()
    evaluator = CHAIR()
    
    # post_anno = post_process_masp_cap_label(evaluator, args.cap_file, args.gt)
    post_anno = process_annotations(evaluator, args.cap_file, args.gt, args.num_sample)
    with open(f"{args.output_file}", "w") as file:
        json.dump(post_anno, file, indent=4)

