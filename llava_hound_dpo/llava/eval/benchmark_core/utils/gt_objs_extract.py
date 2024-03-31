import os
import re
import json
import sys
import argparse
from nltk.stem import *
import nltk
import openai
from abc import ABC, abstractmethod
# from pattern3.en import singularize
from nltk.stem import WordNetLemmatizer
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

    def request(self, system_content, usr_question):
        response = openai.ChatCompletion.create(
            engine="gpt_openapi",
            messages=[
                {"role": "system", "content": f"{system_content}"},
                {"role": "user", "content": f"{usr_question}"}
            ],
            temperature=self.temperature, 
            model=self.model
        )
        resp = response.choices[0]['message']['content']
        total_tokens = response.usage['total_tokens']

        return resp, total_tokens
    
    def get_completion(self, user_prompt=None, system_prompt=None,max_try=10):
        gpt_cv_nlp = '[]'
        key_i = 0
        total_tokens = 0
        # gpt_cv_nlp, total_tokens = self.request(system_prompt, user_prompt)
        while max_try > 0:
            try:
                gpt_cv_nlp, total_tokens = self.request(system_prompt, user_prompt)
                # print('Succ: ', gpt_cv_nlp)
                max_try = 0
                break
            except:
                print("fail ", max_try)
                key = self.key_pool[key_i%2]
                openai.api_key = key
                key_i += 1
                time.sleep(5)
                max_try -= 1
    
        return gpt_cv_nlp, total_tokens

class CHAIR(object):

    def __init__(self) -> None:
        super().__init__()
        self.system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        self.openai_obj = OpenAIAPIWrapper(key_pool=["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"])
        with open('/mnt/bd/bohanzhaiv0/MASP_PROD/prompts/cap2objs.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt = content
    
    def cap2objs_gpt4(self, cap):
        user_prompt = self.cap_user_prompt.format_map({'cap':cap})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        if match:
            objects_list_str = match.group(1)
            # Split the string into a list of items
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return objects_in_image
        else:
            return []

def post_process_masp_cap_label(evaluator, annotations_file):
    results = []
    annotations = json.load(open(annotations_file))
    for data in tqdm(annotations):
        caption = data['caption']
        cap_objs = evaluator.cap2objs_gpt4(caption)
        data['gt_objs'] = cap_objs
        results.append(data)
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default='/mnt/bn/yukunfeng-nasdrive/xiangchen/dataset/masp/meta_data_v0.json')
    parser.add_argument("--output_file", type=str, default='')
    args = parser.parse_args()
    evaluator = CHAIR()
    
    post_anno = post_process_masp_cap_label(evaluator, args.cap_file)
    with open(f"/mnt/bd/bohanzhaiv0/MASP_PROD/benchmark_data/{args.output_file}", "w") as file:
            json.dump(post_anno, file, indent=4)


