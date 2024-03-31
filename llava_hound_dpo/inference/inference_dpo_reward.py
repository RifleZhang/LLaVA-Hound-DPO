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
import copy
from dataclasses import dataclass, field
from trl.trainer.utils import DPODataCollatorWithPadding

from typing import Dict, Optional, Sequence, List, Any, Tuple, Union
import transformers
from torch.utils.data import Dataset, DataLoader
from llava.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

from data_processing.utils import load_json_data, save_json, save_jsonl
from dpo_scripts.run_dpo import preprocess

from transformers import CLIPImageProcessor
from PIL import Image

def get_id_from_path(path):
    return path.split('/')[-1].split('.')[0]

MODAL_TOKEN_LIST=["<video>", "<image>"]
def remove_special_tokens(text):
    for token in MODAL_TOKEN_LIST:
        if token in text:
            text = text.replace(token, "").strip()
    return text

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

def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]

def maybe_trucate(text, max_len=256):
    n_words = len(text.split())
    if n_words > max_len:
        text = " ".join(text.split()[:max_len])
    return text

@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    def collate(self, batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("input_ids")  or k.endswith("labels"):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue
            
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value).to("cuda")
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        
        padded_batch["attention_mask"] = padded_batch["input_ids"].ne(self.tokenizer.pad_token_id).to("cuda")
        return padded_batch

    def tokenize_batch_element(
        self,
        prompt: str,
        answer: str,
        has_X: str = None
    ) -> Dict:
        
        sources = make_conv(prompt, answer)
        chosen_data_dict = preprocess([sources], self.tokenizer, has_X=has_X)
        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        return chosen_data_dict
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        Xs, keys = [], []
        ids = []
        has_X = 'video'
        for feature in features:
            prompt = feature["prompt"]
            answer = maybe_trucate(feature["model_prediction"]["message"])
            ids.append(feature['id'])
            Xs.append(feature[has_X])
            keys.append(has_X)
             
            batch_element = self.tokenize_batch_element(prompt, answer, has_X=has_X)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch =  self.collate(tokenized_batch)
        padded_batch['images'] = [Xs, keys]  # we do not change the key's name, tho they are video frames
        padded_batch['ids'] = ids
        return padded_batch
    
class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data, video_processor, video_folder):
        super(Dataset, self).__init__()
        
        self.list_data_dict = data
        self.video_processor = video_processor
        self.video_folder = video_folder

    def __len__(self):
        # return 20
        return len(self.list_data_dict)

    def __getitem__(self, i):
        '''
        {'id': '10003718_0_0',
            'query': 'What is the color of the water in the video?',
            'answer': 'The color of the water in the video appears to be a murky blue-gray due to the sediment.',
            'modal_type': 'VIDEO',
            'video_decode_backend': 'frames',
            'temperature': 1.0,
            'top_p': 0.95,
            'max_new_tokens': 1024,
            'model_prediction': {'status': 'success',
            'message': 'The water in the video has a muddy brown color.'},
            'video': 'test/webvid/10003718'}
        '''
        
        # sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = copy.deepcopy(self.list_data_dict[i]) # inplace modification following
        

        processor = self.video_processor
        video = data_dict['modal_path']
        # video_file = data_dict['video']
        # video_folder = self.video_folder
        # video = os.path.join(video_folder, video_file)
        video = processor(video, return_tensors='pt')['pixel_values'][0].half().to('cuda')
 
        # sources = preprocess_multimodal(make_conversation([e["detail"] for e in sources]), self.data_args)
        prompt = data_dict['query']
        prompt = prompt.replace("<video>", "").strip()
        prompt = "<video>\n" + prompt
        data_dict['prompt'] = prompt        
        data_dict['video'] = video
        
        return data_dict

def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


@torch.no_grad()
def compute_logp(model, batch):
    all_logits, new_labels = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            images=batch["images"],
            use_cache=False,
            dpo_forward=True,
        )
    all_logits = all_logits.to(torch.float32)
    all_logps = get_batch_logps(
        all_logits,
        new_labels,
        average_log_prob=False,
        label_pad_token_id=-100,
    )
    return all_logps

@torch.no_grad()
def main(model_path, reference_model_path, data_path, video_dir, output_dir, output_name, chunk_idx, chunks, **kwargs):    
    data = load_json_data(data_path)
    chunk_data = get_chunk(data, chunks, chunk_idx)
    # chunk_data = chunk_data[:32]

    save_path = f"{output_dir}/{output_name}"
    if os.path.exists(save_path):
        ll = len(chunk_data)
        res = load_json_data(save_path)
        all_ids = []
        for rr in res:
            all_ids.extend([r['id'] for r in rr])
        all_ids = set(all_ids)
        chunk_data = [d for d in chunk_data if d['id'] not in all_ids]
        logger.info(f"load {len(res)}, full chunck length: {ll}, need process length: {len(chunk_data)}")
    fout = open(save_path, 'a')
   
    model_name = get_model_name_from_path(model_path)
    logger.info(f"model {model_name}")
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, device_map={"":0}, use_flash_attn=True)
    _, ref_model, _, _ = load_pretrained_model(reference_model_path, None, model_name, device_map={"":0}, use_flash_attn=True)
    eval_dataset = DPODataset(data=chunk_data,
                              video_processor = processor['video'], 
                              video_folder = video_dir,)
    data_collator = DPODataCollator(
            tokenizer = tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    dataloader = DataLoader(dataset=eval_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

 
    for batch in tqdm(dataloader, total=len(dataloader)):
        chosen_logps = compute_logp(model, batch)
        reference_chosen_logps = compute_logp(ref_model, batch)

        reward = chosen_logps - reference_chosen_logps
        reward = reward.cpu().numpy().tolist()
        ids = batch['ids']
        cur_result = [{"id": i, "reward": r} for i, r in zip(ids, reward)]
        fout.write(json.dumps(cur_result) + '\n')

if __name__ == "__main__":
    fire.Fire(main)