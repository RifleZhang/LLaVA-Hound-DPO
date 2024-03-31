from audioop import avg
from email.policy import default
import os
import re
import json
import sys
import argparse

import openai
from abc import ABC, abstractmethod
# from pattern3.en import singularize
# from nltk.stem import WordNetLemmatizer
# from call_dino_service import 
from tqdm import tqdm
from functools import partial

# import spacy
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool

# Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")

# 0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT 
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
        # openai.api_base = "https://search-sg.bytedance.net/gpt/openapi/online/v2/crawl"
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
    
    def get_completion(self, user_prompt=None, system_prompt=None,max_try=20):
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
                # key = self.key_pool[key_i%2]
                # openai.api_key = key
                key_i += 1
                time.sleep(2)
                max_try -= 1
    
        return gpt_cv_nlp, total_tokens

class RefineCHAIR(object):
    def __init__(self):
        self.system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        self.openai_obj = OpenAIAPIWrapper(key_pool=["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"])
        with open('/mnt/bn/algo-masp-nas-2/ruohongz/evaluation/MASP_PROD_2/prompts/cap_mention.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt = content

    def add(self, case_res, all_res):
        for key, value in case_res.items():
            for idx, count_ in enumerate(value):
                all_res[key][idx] += count_
        return
    
    def save_metric(self, coverage, hallucination, case_len, output_dir=None):
        final_metrics = {}
        for name, res in [['coverage', coverage], ['hallucination', hallucination]]:
            combine_counter = [0, 0]    
            for cat, counter in res.items():
                final_metrics[name+'_'+cat] = round(counter[0] * 100/ counter[1], 2)
                combine_counter[0] += counter[0]
                combine_counter[1] += counter[1]
                if name == 'hallucination':
                    final_metrics[name+'_'+cat] = round(100 - final_metrics[name+'_'+cat], 2)
                final_metrics[name] = round(combine_counter[0] * 100 / combine_counter[1], 2)
            if name == 'hallucination':
                final_metrics[name] = round(100 - final_metrics[name], 2)
        final_metrics['avg_len'] = round(sum(case_len) / len(case_len), 1)

        if output_dir is not None:
            with (output_dir / 'chair_metric.json').open('w') as f:
                json.dump(final_metrics, f, indent=4)

        print(json.dumps(final_metrics, indent=1))

    def combine_info(self, pred_info, gt_info):
        combined_info = defaultdict(dict)
        for gt in gt_info:
            object_id = gt['object_id']
            if gt['cap_info'] is None:
                continue
            combined_info[object_id]['gt_caption'] = gt['refine_caption']
            combined_info[object_id]['gt_info'] = gt['cap_info']

        for pred in pred_info:
            object_id = pred['object_id']
            if object_id not in combined_info:
                # print(pred)
                continue
            if pred['cap_info'] is None:
                continue
            combined_info[object_id]['pred_caption'] = pred['masp_inference']
            combined_info[object_id]['pred_info'] = pred['cap_info']
        filtered_ids = []
        for key, value in combined_info.items():
            if ('pred_info' not in value) or ('gt_info' not in value):
                filtered_ids.append(key)
        for obj_id in filtered_ids:
            del combined_info[obj_id]

        print(f'evaluation cases: {len(combined_info)}')
        
        return combined_info

    def format_question(self, info):
        categories = ['subjects', 'activities', 'locations', 'text_overlays']
        question_id = 0
        question_mapping = {}
        questions = []
        for cat in categories:
            if cat == 'subjects':
                for c_id, character_content in enumerate(info['subjects']):
                    questions.append(cat + ':' + character_content['name'])
                    question_mapping[question_id] = (cat, c_id)
                    question_id += 1
                    if 'attributes' not in character_content:
                        continue
                    for a_id, attr in enumerate(character_content['attributes']):
                        questions.append(character_content['name'] + ':' + attr)
                        question_mapping[question_id] = ('attributes', c_id, a_id)
                        question_id += 1
                
            else:
                for c_id, cat_attr in enumerate(info[cat]):
                    questions.append(cat + ':' + cat_attr)
                    question_mapping[question_id] = (cat, c_id)
                    question_id += 1
                    
        question_str = ''
        for idx, q in enumerate(questions):
            question_str += f'{idx+1}. {q}' + '\n'

        return question_str, question_mapping
    
    def parsing_results(self, gpt_ret, question_mapping):
        gpt_ret = gpt_ret.lower()
        pattern = r'(\d+)\.(.+) - (yes|no|maybe),(.+)'

        # Find all matches in the text
        matches = re.findall(pattern, gpt_ret)
        collected_answer = defaultdict(lambda:[0,0])
        # Print the matches
        for match in matches:
            question_id, question, answer, reason = match
            question_id = int(question_id) - 1
            cat = question_mapping[question_id][0]
            collected_answer[cat][1] += 1
            if 'yes' in answer:
                collected_answer[cat][0] += 1
            elif 'no' in answer:
                pass
            elif 'maybe' in answer:
                collected_answer[cat][0] += 1
            else:
                NotImplementedError
        return collected_answer



def process_coverage(data, evaluator):
    object_id = data[0]
    case_info = data[1]
    gt_info = case_info['gt_info']
    # if gt_info is None:
    #     return None
    try:
        question_str, question_mapping = evaluator.format_question(gt_info)
    except Exception as e:
        print(e)
        return None
    user_prompt = deepcopy(evaluator.cap_user_prompt)
    user_prompt = user_prompt.replace("/video caption/", case_info['pred_caption'])
    user_prompt = user_prompt.replace("/question/", question_str)
    gpt_ret, total_tokens = evaluator.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=evaluator.system_prompt,max_try=20)
    coverage_res = evaluator.parsing_results(gpt_ret, question_mapping)
    sentence_len = len(case_info['pred_caption'].split(' '))
    return (object_id, gpt_ret, dict(coverage_res), sentence_len)


def process_hallucination(data, evaluator):
    object_id = data[0]
    case_info = data[1]
    pred_info = case_info['pred_info']
    # if pred_info is None:
    #     return None
    try:
        question_str, question_mapping = evaluator.format_question(pred_info)
    except Exception as e:
        print(e)
        return None
    user_prompt = deepcopy(evaluator.cap_user_prompt)
    user_prompt = user_prompt.replace("/video caption/", case_info['gt_caption'])
    user_prompt = user_prompt.replace("/question/", question_str)
    gpt_ret, total_tokens = evaluator.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=evaluator.system_prompt,max_try=20)
    hallucination_res = evaluator.parsing_results(gpt_ret, question_mapping)
    # self._add(hallucination_res, evaluator.hallucination_metric)
    # saved_combined_info[object_id]['hallucination_res'] = gpt_ret
    # print(gpt_ret)    
    return (object_id, gpt_ret, dict(hallucination_res))



def compute_refine_chair(pred_file, gt_file, evaluator):
    coverage_metric = defaultdict(lambda:[0,0])
    hallucination_metric = defaultdict(lambda:[0,0])
    case_len = []

    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_info = json.load(f)
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_info = json.load(f)

    combined_info = evaluator.combine_info(pred_info, gt_info)
    saved_combined_info = deepcopy(combined_info) 
    combine_info_lst = list(combined_info.items())
    process_coverage_partial = partial(process_coverage, evaluator=evaluator)
    process_hallucination_partial = partial(process_hallucination, evaluator=evaluator)
    combine_info_lst = list(combined_info.items())
    for key, case_info in combine_info_lst:
        gt_info = case_info['gt_info']
        pred_info = case_info['pred_info']
        evaluator.format_question(gt_info)
        evaluator.format_question(pred_info)
       
    pool = Pool(processes=8)
    print('calculate coverage')
    for res in tqdm(pool.imap_unordered(process_coverage_partial, combine_info_lst), total=len(combine_info_lst)):
        if res is None:
            continue
        object_id, gpt_ret, coverage_res, sentence_len = res
        evaluator.add(coverage_res, coverage_metric)
        case_len.append(sentence_len)
        saved_combined_info[object_id]['coverage_res'] = gpt_ret
       
    print('calculate hallucination')

    for res in tqdm(pool.imap_unordered(process_hallucination_partial, combine_info_lst), total=len(combine_info_lst)):
        if res is None:
            continue
        object_id, gpt_ret, hallucination_res = res
        evaluator.add(hallucination_res, hallucination_metric)
        saved_combined_info[object_id]['hallucination_res'] = gpt_ret
    
    pool.close()
    pool.join()

    output_dir = Path(pred_file).parent
    evaluator.save_metric(coverage_metric, hallucination_metric, case_len, output_dir)
    with (output_dir / 'chair_metric_detailed_res.json').open('w') as f:
        json.dump(saved_combined_info, f, indent=4)     

    
def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    metric_string = "%0.01f\t%0.01f" %(sentence_metrics['CHAIRs']*100, 
                                       sentence_metrics['CHAIRi']*100)
    if not quiet:
        print("CHAIRs\tCHAIRi")
        print(metric_string)
        print(sentence_metrics['sentence len'])
        print(sentence_metrics['avg objects'])
    else:
        return metric_string

    
# python3 chair/chair_gpt.py --cap_file /mnt/bd/bohanzhaiv1/LLM/bohan/POPE/caption_data/vg_instruction1_llava.json  --annotation_path /mnt/bn/algo-masp-nas-2/masp_data/coco_2014/annotations
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default='/mnt/bn/yukunfeng-nasdrive/xiangchen/model/masp_models/checkpoints/mistral-m3it-ttvqa_single-7k-sharegpt4v-uniform/video_chair/vid_top1k_res_info.json')
    parser.add_argument("--gt_file", type=str, default='/mnt/bn/yukunfeng-nasdrive/xiangchen/repo/benchmark_data/refine_chair_eval_gt.json')
    args = parser.parse_args()

    evaluator = RefineCHAIR()
    compute_refine_chair(args.pred_file, args.gt_file, evaluator)
