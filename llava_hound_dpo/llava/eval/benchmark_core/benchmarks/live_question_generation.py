import openai
import json
from tqdm import tqdm
import re
import argparse
from chair import OpenAIAPIWrapper


class Live(object):
    def __init__(self):
        self.system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."

        # nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM
        # 0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT
        # self.openai_obj = OpenAIAPIWrapper(key_pool=["0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT", "nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM"])
        self.openai_obj = OpenAIAPIWrapper(key_pool=["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"])
        # self.openai_obj = OpenAIAPIWrapper(key_pool=["nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM"])
        with open('/mnt/bd/bohanzhaiv0/MASP_PROD/prompts/VQA_generate/Live/yesquestion.txt', 'r') as file:
            content = file.read()
        self.yes_prompt = content
        with open('/mnt/bd/bohanzhaiv0/MASP_PROD/prompts/VQA_generate/Live/noquestion.txt', 'r') as file:
            content = file.read()
        self.no_prompt = content
    
    def get_yes_question(self, gt_cap):
        system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        user_prompt = self.yes_prompt.format_map({'cap':gt_cap})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,max_try=10)
        gpt_ret = [line.strip().split('.')[1] for line in gpt_ret.split('\n') if gpt_ret.strip()]
        gpt_ret = list(set(gpt_ret))
        # if 'mismatch' in gpt_ret:
        #     match = re.search(r"mismatch = \[(.*?)\]", gpt_ret)
        # elif 'Mismatch' in gpt_ret:
        #     match = re.search(r"Mismatch = \[(.*?)\]", gpt_ret)
        return gpt_ret
    def get_no_question(self, gt_cap):
        system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        user_prompt = self.no_prompt.format_map({'cap':gt_cap})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,max_try=10)
        gpt_ret = [line.strip().split('.')[1] for line in gpt_ret.split('\n') if gpt_ret.strip()]
        gpt_ret = list(set(gpt_ret))

        # Sort the questions if you need them in order
        # if 'mismatch' in gpt_ret:
        #     match = re.search(r"mismatch = \[(.*?)\]", gpt_ret)
        # elif 'Mismatch' in gpt_ret:
        #     match = re.search(r"Mismatch = \[(.*?)\]", gpt_ret)
        return gpt_ret

def post_process_masp_cap_label(evaluator, annotations_file, output_file, history=None):
    results = history if history else []
    task_ids = set([result['task_id'] for result in results])
    annotations = json.load(open(annotations_file))
    for data in tqdm(annotations):
        if data['task_id'] in task_ids:
            continue
        caption = data['caption']
        yes_question = evaluator.get_yes_question(caption)
        no_question = evaluator.get_no_question(caption)
        data['live_yes'] = yes_question
        data['live_no'] = no_question
        results.append(data)
        with open(f"/mnt/bd/bohanzhaiv0/MASP_PROD/benchmark_data/{output_file}", "w") as file:
            json.dump(results, file, indent=4)
    return results
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default='/mnt/bn/yukunfeng-nasdrive/xiangchen/dataset/masp/meta_data_v0.json')
    parser.add_argument("--output_file", type=str, default='')
    args = parser.parse_args()
    evaluator = Live()
    history = json.load(open("/mnt/bd/bohanzhaiv0/MASP_PROD/benchmark_data/live_vqa.json"))
    resutls = post_process_masp_cap_label(evaluator, args.cap_file, args.output_file, history)
    with open(f"/mnt/bd/bohanzhaiv0/MASP_PROD/benchmark_data/{args.output_file}", "w") as file:
            json.dump(resutls, file, indent=4)
