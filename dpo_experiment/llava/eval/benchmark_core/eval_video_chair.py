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
# from nltk.stem import WordNetLemmatizer
# from call_dino_service import
from tqdm import tqdm

# import spacy
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

    def get_completion(self, user_prompt=None, system_prompt=None, max_try=10):
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
                time.sleep(2)
                max_try -= 1

        return gpt_cv_nlp, total_tokens


class CHAIR(object):
    def __init__(self):
        self.system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."

        # nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM
        # 0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT
        # self.openai_obj = OpenAIAPIWrapper(key_pool=["0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT", "nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM"])
        self.openai_obj = OpenAIAPIWrapper(key_pool=["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"])
        # self.openai_obj = OpenAIAPIWrapper(key_pool=["nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM"])
        self.resource_dir = '/mnt/bn/baiyi-arnold-nas/workspace/mllm_colosseo/llava/eval/resource'
        with open(f'{self.resource_dir}/region_cap2obj_prompt.txt', 'r') as file:
            content = file.read()
        self.region_user_prompt = content
        with open(f'{self.resource_dir}/cap2obj_prompt.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt = content

        with open(f'{self.resource_dir}/hallucination_prompt.txt', 'r') as file:
            content = file.read()
        self.hall_user_prompt = content

        with open(f'{self.resource_dir}/coverage_prompt.txt', 'r') as file:
            content = file.read()
        self.coverage_user_prompt = content

        # read in synonyms
        synonyms = open('/mnt/bn/algo-masp-nas-2/benchmark/CHAIR/Hallucination/data/synonyms.txt').readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
                             'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case',
                             'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog',
                             'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie',
                             'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']

        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal',
                        'cub']
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'


    def list_region2cap(self, list_regions):
        system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information."
        user_prompt = self.region_user_prompt.format_map({'list_of_regions': list(set(list_regions))})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,
                                                               max_try=10)
        print('region result ', gpt_ret, total_tokens)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        if match:
            objects_list_str = match.group(1)
            # Split the string into a list of items
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return list(set(objects_in_image))
        else:
            return []

    def cap2objs_gpt4(self, cap):
        system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        user_prompt = self.cap_user_prompt.format_map({'cap': cap})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,
                                                               max_try=10)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        if match:
            objects_list_str = match.group(1)
            # Split the string into a list of items
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return objects_in_image
        else:
            return []

    def cap2objs_spacy(self, cap):
        return []


    def get_hall_gpt4(self, gt, cap_obj):
        system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."
        user_prompt = self.hall_user_prompt.format_map({'gt': gt, 'cap_obj': cap_obj})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,
                                                               max_try=10)
        # if 'mismatch' in gpt_ret:
        #     match = re.search(r"mismatch = \[(.*?)\]", gpt_ret)
        # elif 'Mismatch' in gpt_ret:
        #     match = re.search(r"Mismatch = \[(.*?)\]", gpt_ret)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        print(gpt_ret)
        print('Match is ', match)
        print('gt is: ', gt)
        print('cap_obj is: ', cap_obj)
        if match:
            objects_list_str = match.group(1)
            # Split the string into a list of items
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return list(set(objects_in_image))
        else:
            return []

    def get_uncover_gpt4(self, gt, cap_obj):
        user_prompt = self.coverage_user_prompt.format_map({'gt': gt, 'cap_obj': cap_obj})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt,
                                                               system_prompt=self.system_prompt, max_try=10)

        match = re.search(r"\[(.*?)\]", gpt_ret)
        print(gpt_ret)
        print('Not covered words are ', match)
        print('gt is: ', gt)
        print('cap_obj is: ', cap_obj)
        if match:
            objects_list_str = match.group(1)
            # Split the string into a list of items
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return list(set(objects_in_image))
        else:
            return []

    def are_similar(self, a, b):
        return a.lower() == b.lower()

    def get_uncover(self, list_A, list_B):
        """
            x in B such that x is not in A.
            A = prediction, B = ground truth, then coverage is similar to recall.
            A = ground truth, B = prediction, then coverage is similar to precision.
        """
        uncover = []

        for item_b in list_B:
            found = False
            for item_a in list_A:
                if self.are_similar(item_a, item_b):
                    found = True
                    break
            if not found:
                uncover.append(item_b)

        return uncover

    def compute_chair_video(self, eval_file, num_sample=None):
        video_infos = json.load(open(eval_file))
        if num_sample is not None:
            video_infos = video_infos[:num_sample]
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        output = {'sentences': []}
        avg_len = 0
        all_output = []

        for i, eval_Data in tqdm(enumerate(video_infos), total=len(video_infos)):
            # cap = eval_Data['otter_inference']
            cap = eval_Data['masp_inference']
            raw_words = self.cap2objs_gpt4(cap)
            gt_objects = eval_Data['obj_info']['gt_objs']
            hallucinated_words = self.get_hall_gpt4(gt_objects, raw_words)
            hallucinated_words = [item for item in hallucinated_words if item != '']
            cap = cap.strip()
            sent_len = len(cap.split(' '))
            avg_len += sent_len

            coco_word_count += len(raw_words)
            print('total len: ', coco_word_count)
            print('cap items len: ', len(raw_words))
            print('hallucinated words len: ', len(hallucinated_words), hallucinated_words, type(hallucinated_words))
            hallucinated_word_count += len(hallucinated_words)
            print('total hall is ', hallucinated_word_count)
            num_caps += 1
            if len(hallucinated_words) > 0:
                num_hallucinated_caps += 1

            out = {
                "raw_words": raw_words,
                "hallucinated_words": hallucinated_words,
                "sent_len": sent_len,
                "cap_len": len(raw_words),
                "hallucinated_word_count": len(hallucinated_words)
            }
            all_output.append(out)

        chair_s = (num_hallucinated_caps / num_caps)
        chair_i = (hallucinated_word_count / coco_word_count)
        output['overall_metrics'] = {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i,
            'sentence len': avg_len / num_caps,
            'avg objects': coco_word_count / num_caps
        }
        print(output)
        with open(f"{eval_file}.info.json", "w") as file:
            json.dump(all_output, file, indent=4)

        return output


def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    metric_string = "%0.01f\t%0.01f" % (sentence_metrics['CHAIRs'] * 100,
                                        sentence_metrics['CHAIRi'] * 100)
    if not quiet:
        print("CHAIRs\tCHAIRi")
        print(metric_string)
        print(sentence_metrics['sentence len'])
        print(sentence_metrics['avg objects'])
    else:
        return metric_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default='')
    parser.add_argument("--uncover", type=bool, default=False)
    parser.add_argument("--num_sample", type=int, default=None)
    args = parser.parse_args()

    evaluator = CHAIR()

    cap_dict = evaluator.compute_chair_video(args.cap_file, num_sample=args.num_sample)
    print_metrics(cap_dict)




