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

# import spacy
import time

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
    
    def get_completion(self, user_prompt=None, system_prompt=None,max_try=30):
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

def combine_coco_captions(annotation_path):

    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}
    return all_caps

def combine_coco_instances(annotation_path):

    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}
    return all_instances

def objects2name(obj):
    image_id = obj['image_id']
    objects = obj['objects']
    object_names = [single_obj['name'][0] for single_obj in objects]
    return object_names

def region2summarize(region):
    image_id = region['id']
    regions = region['regions']
    region_sents = [reg['phrase'] for reg in regions]
    return region_sents


def combine_vg_instances(vg_path, num=1000):
    images_info = {}
    objects = json.load(open('%s/objects.json' %(vg_path)))[:num]
    attributes = json.load(open('%s/attributes.json' %(vg_path)))[:num]
    region_des = json.load(open('%s/region_descriptions.json' %(vg_path)))[:num]
    for obj, attr, region in zip(objects, attributes, region_des):
        assert obj['image_id'] == attr['image_id'] == region['id']
        image_id = obj['image_id']
        images_info[image_id] = {}
        images_info[image_id]['objects'] = obj
        images_info[image_id]['bbox_objects'] = list(set([o['names'][0] for o in obj['objects']]))
        images_info[image_id]['attributes'] = attr
        images_info[image_id]['regions'] = region
        images_info[image_id]['regions_summary'] = region2summarize(region)
    
    return images_info





class CHAIR(object):
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self.system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information, I can't guarantee perfection, and it's always a good idea to consult additional resources or professionals when making critical decisions based on the information I provide."

        # nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM
        # 0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT
        # self.openai_obj = OpenAIAPIWrapper(key_pool=["0TYWZPqumqoWSVdkoJCb5W5P9lEqjUKT", "nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM"])
        self.openai_obj = OpenAIAPIWrapper(key_pool=["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"])
        # self.openai_obj = OpenAIAPIWrapper(key_pool=["nlCj9fvmLyJIYBMhB9jmFgStVaT8XWEM"])
        with open('/mnt/bd/bohanzhaiv0/LLM/POPE/chair/region_cap2obj_prompt.txt', 'r') as file:
            content = file.read()
        self.region_user_prompt = content
        # /mnt/bd/bohanzhaiv1/LLM/bohan/POPE/chair/cap2obj_prompt.txt
        # /mnt/bd/bohanzhaiv1/LLM/bohan/POPE/chair/cap2obj_prompt_bracket.txt
        with open('/mnt/bd/bohanzhaiv0/MASP_PROD/prompts/cap2objs.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt = content

        with open('/mnt/bd/bohanzhaiv0/LLM/POPE/chair/hallucination_prompt.txt', 'r') as file:
            content = file.read()
        self.hall_user_prompt = content

        with open('/mnt/bd/bohanzhaiv0/MASP_PROD/prompts/object_coverage.txt', 'r') as file:
            content = file.read()
        self.coverage_user_prompt = content

        #read in synonyms
        synonyms = open('/mnt/bn/algo-masp-nas-2/benchmark/CHAIR/Hallucination/data/synonyms.txt').readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]
        
        #Some hard coded rules for implementing CHAIR metrics on MSCOCO
        
        #common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
    
    
    
    def list_region2cap(self, list_regions):
        system_prompt = "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model. I'm designed to understand and generate human-like text based on the input I receive. My main purpose is to assist with information, answer questions, help with tasks that involve natural language processing, and engage in conversations with users.Please note that while I aim to provide accurate and reliable information."
        user_prompt = self.region_user_prompt.format_map({'list_of_regions':list(set(list_regions))})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,max_try=10)
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
        user_prompt = self.cap_user_prompt.format_map({'cap':cap})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,max_try=10)
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
        user_prompt = self.hall_user_prompt.format_map({'gt':gt, 'cap_obj':cap_obj})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=system_prompt,max_try=10)
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
    
    def get_annotations(self):
        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''
        self.get_annotations_from_segments() 
        self.get_annotations_from_captions()
# python3 chair_gpt.py --cap_file /mnt/bd/bohanzhaiv1/LLM/bohan/POPE/caption_data/Instruction1_llava.json --annotation_path /mnt/bn/algo-masp-nas-2/masp_data/coco_2014/annotations

    def get_uncover_gpt4(self, gt, cap_obj):
        user_prompt = self.coverage_user_prompt.format_map({'cap_obj':cap_obj, 'gt':gt})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)

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


    def converage(self, cap_file, vg_path):
        image_infos = json.load(open('/mnt/bd/bohanzhaiv0/LLM/POPE/chair/vg_info_100.json'))
        num_caps = 0.
        caps = json.load(open(cap_file))
        caps = caps[:100]

        output = {'sentences': []} 
        avg_len = 0
        coco_word_count = 0.
        uncover_word_counts = 0
        num_uncovered_count = 0
        sent_objects = 0

        for i, cap_eval in tqdm(enumerate(caps), total=len(caps)):
            cap = cap_eval['text']
            print('caption is ', cap)
            imid = cap_eval['image_id']
            if str(i+1) in image_infos or (i+1 in image_infos):
                # image_summary = image_infos[str(i+1)]['image_summary']
                gt_objects = image_infos[str(i+1)]['gt_objs']
            else:
                exit()
            raw_words = self.cap2objs_gpt4(cap)
            param_words = re.findall(r'\[(.*?)\]', cap)
            # param_words = []
            # raw_words = raw_words # + param_words
            # raw_words = param_words
            raw_words = list(set(raw_words[:] + param_words))
            gt_objects = list(set(gt_objects))

            uncover_words = self.get_uncover_gpt4(gt_objects, raw_words)
            uncover_words = [item for item in uncover_words if item != '']
            sent_len = len(cap.split(' '))
            avg_len += sent_len
            print('image id: ', imid)
            print('sent len is ', sent_len)

            sent_objects += len(raw_words)
            coco_word_count += len(gt_objects) 
            print('total len: ', coco_word_count)
            print('cap items len: ', len(raw_words))
            print('uncovered words len: ', len(uncover_words), uncover_words, type(uncover_words))
            uncover_word_counts += len(uncover_words)
            print('total uncovered word is ', uncover_word_counts)
            print('coverage is ', (coco_word_count-uncover_word_counts)/coco_word_count)
            num_caps += 1
            if len(uncover_words) > 0:
                num_uncovered_count += 1
        
        uncover_s = (num_uncovered_count/num_caps)
        uncover_i = (uncover_word_counts/coco_word_count)
        output['overall_metrics'] = {
                                     'Uncovers': uncover_s,
                                     'Uncoveri': uncover_i,
                                     'Coveri': 1 - uncover_i,
                                     'sentence len':avg_len / num_caps,
                                     'avg gt objects': coco_word_count / num_caps,
                                     'avg cap objects': sent_objects / num_caps
                                     }
        return output
    def compute_chair_video_coverage(self, eval_file):
        video_infos = json.load(open(eval_file))
        num_caps = 0.
        output = {'sentences': []} 
        avg_len = 0
        coco_word_count = 0.
        uncover_word_counts = 0
        num_uncovered_count = 0
        sent_objects = 0

        for i, eval_Data in tqdm(enumerate(video_infos), total=len(video_infos)):
            # cap = eval_Data['otter_inference']
            cap = eval_Data['masp_inference']
            raw_words = self.cap2objs_gpt4(cap)
            gt_objects = eval_Data['obj_info']['gt_objs']
            uncover_words = self.get_uncover_gpt4(gt_objects, raw_words)
            uncover_words = [item for item in uncover_words if item != '']
            sent_len = len(cap.split(' '))
            avg_len += sent_len
            print('sent len is ', sent_len)

            sent_objects += len(raw_words)
            coco_word_count += len(gt_objects) 
            print('total len: ', coco_word_count)
            print('cap items len: ', len(raw_words))
            print('uncovered words len: ', len(uncover_words), uncover_words, type(uncover_words))
            uncover_word_counts += len(uncover_words)
            print('total uncovered word is ', uncover_word_counts)
            print('coverage is ', (coco_word_count-uncover_word_counts)/coco_word_count)
            num_caps += 1
            if len(uncover_words) > 0:
                num_uncovered_count += 1
        
        uncover_s = (num_uncovered_count/num_caps)
        uncover_i = (uncover_word_counts/coco_word_count)
        output['overall_metrics'] = {
                                     'Uncovers': uncover_s,
                                     'Uncoveri': uncover_i,
                                     'Coveri': 1 - uncover_i,
                                     'sentence len':avg_len / num_caps,
                                     'avg gt objects': coco_word_count / num_caps,
                                     'avg cap objects': sent_objects / num_caps
                                     }
        return output


# user_prompt = f"I have an object list {objects_in_image} we call it list A, and a model captured an object list {obj_list} we call it list B, if list B's object or similar meaning but not same word occurred in list A, we called it a match, if either same words, synonyms or similar meaning object not occurred in list A we call it mismatch, plz return a list of mismatch. the output should be a list like mismatch = ['human', 'beach'], I don't a code return your answer"
    def compute_chair_vg(self, cap_file, vg_path):
        # self._load_generated_captions_into_evaluator(cap_file)
        # image_infos = combine_vg_instances(vg_path)
        image_infos = json.load(open('/mnt/bd/bohanzhaiv0/LLM/POPE/chair/vg_info_100.json'))
        # image_processed = json.load(open('/mnt/bd/bohanzhaiv1/LLM/bohan/POPE/chair/vg_info_10.json'))
        # # image_processed = {}
        # for i in tqdm(range(100)):
        #     print(i)
        #     if str(i+1) in image_processed:
        #         continue
        #     image_summary = self.list_region2cap(image_infos[i+1]['regions_summary'])
        #     print(image_summary)
        #     image_infos[i+1]['image_summary'] = list(set(image_summary))
        #     # gt_objects = self.cap2objs_gpt4(image_summary)
        #     gt_objects = list(set(image_summary +image_infos[i+1]['bbox_objects']))
        #     image_infos[i+1]['gt_objs'] = gt_objects
        #     image_processed[i+1] = image_infos[i+1]
        #     with open("/mnt/bd/bohanzhaiv1/LLM/bohan/POPE/chair/vg_info_100.json", "w") as file:
        #         json.dump(image_processed, file, indent=4)
        

        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        caps = json.load(open(cap_file))
        caps = caps[:100]

        output = {'sentences': []} 

        avg_len = 0
    
        for i, cap_eval in tqdm(enumerate(caps), total=len(caps)):
            cap = cap_eval['text']
            print('caption is ', cap)
            imid = cap_eval['image_id']
            if str(i+1) in image_infos or (i+1 in image_infos):
                # image_summary = image_infos[str(i+1)]['image_summary']
                gt_objects = image_infos[str(i+1)]['gt_objs']
            else:
                exit()
            raw_words = self.cap2objs_gpt4(cap)
            # raw_words = re.findall(r'\[(.*?)\]', cap)

            # image_summary = self.list_region2cap(image_infos[i+1]['regions_summary'])
            # image_infos[i+1]['image_summary'] = image_summary
            # gt_objects = self.cap2objs_gpt4(image_summary)
            # image_infos[i+1]['gt_objs'] = gt_objects
            # raw_words = self.cap2objs_gpt4(cap)

            hallucinated_words = self.get_hall_gpt4(gt_objects, raw_words)
            hallucinated_words = [item for item in hallucinated_words if item != '']
            sent_len = len(cap.split(' '))
            avg_len += sent_len
            print('image id: ', imid)
            print('sent len is ', sent_len)
            # print(hallucinated_words)

            cap_dict = {'image_id': cap_eval['image_id'], 
                        'caption': cap,
                        'mscoco_hallucinated_words': hallucinated_words,
                        'mscoco_gt_words': list(gt_objects),
                        'hallucination_idxs': [], 
                        'words': raw_words 
                        }

            coco_word_count += len(raw_words) 
            print('total len: ', coco_word_count)
            print('cap items len: ', len(raw_words))
            print('hallucinated words len: ', len(hallucinated_words), hallucinated_words, type(hallucinated_words))
            hallucinated_word_count += len(hallucinated_words)
            print('total hall is ', hallucinated_word_count)
            num_caps += 1
            if len(hallucinated_words) > 0:
                num_hallucinated_caps += 1
        
        with open("/mnt/bd/bohanzhaiv0/LLM/POPE/chair/vg_sm_obj_info.json", "w") as file:
            json.dump(image_infos, file, indent=4)
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        output['overall_metrics'] = {
                                     'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'sentence len':avg_len / num_caps,
                                     'avg objects': coco_word_count / num_caps
                                     }
    
        return output
            
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
    parser.add_argument("--cap_file", type=str, default='')
    parser.add_argument("--uncover", type=bool, default=False)
    parser.add_argument("--sm", type=bool, default=False)
    parser.add_argument("--annotation_path", type=str, default='/mnt/bn/algo-masp-nas-2/masp_data/coco_2014/annotations')
    args = parser.parse_args()


    evaluator = CHAIR(args.annotation_path) 
    # evaluator.get_annotations()

    if not args.uncover:
        cap_dict = evaluator.compute_chair_vg(args.cap_file, '/mnt/bn/data-tns-algo-masp/data/VisualGenome_task') 
        print_metrics(cap_dict)
    else:
        uncover_dict = evaluator.compute_chair_video_coverage(args.cap_file)
        print(uncover_dict)