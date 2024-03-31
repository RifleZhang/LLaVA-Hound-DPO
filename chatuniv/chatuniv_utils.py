import os
import numpy as np
from PIL import Image
import dataclasses
from enum import auto, Enum
from typing import List
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from transformers import StoppingCriteria

import torch
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.mm_utils import tokenizer_image_token

CACHE_DIR=os.environ.get("CACHE_DIR", None)


def remove_special_tokens(text):
    for token in ["<video>", "<vid_patch>", "<vid_start>", "<vid_end>"]:
        if token in text:
            text = text.replace(token, "").strip()
    return text

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
    
def get_seq_frames(total_num_frames, desired_num_frames):
        """
        Calculate the indices of frames to extract from a video.

        Parameters:
        total_num_frames (int): Total number of frames in the video.
        desired_num_frames (int): Desired number of frames to extract.

        Returns:
        list: List of indices of frames to extract.
        """

        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(total_num_frames - 1) / desired_num_frames

        seq = []
        for i in range(desired_num_frames):
            # Calculate the start and end indices of each segment
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))

            # Append the middle index of the segment to the list
            seq.append((start + end) // 2)

        return seq

def resize_image(input_image, target_size=(224, 224)):
    # Load the image if a path is provided
    if isinstance(input_image, str):
        input_image = Image.open(input_image)
    
    # Resize the image using the LANCZOS filter for high-quality downsampling
    resized_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
    
    return resized_image

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

class ModelInference:
    def __init__(self, model, tokenizer, image_processor):
        self.model=model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def get_frames(self, frame_dir, max_len=100):
        # max_len follow video chatgpt implementation
        frames = []
        files = os.listdir(frame_dir)
        files = sorted(files)
        ll = len(os.listdir(frame_dir))
        if ll > max_len:
            indices = get_seq_frames(ll, max_len)
        else:
            indices = list(range(ll))
        for i in indices:
            frame_path = os.path.join(frame_dir, files[i])
            frame = Image.open(frame_path)
            frames.append(frame)
        # Set target image height and width
        target_h, target_w = 224, 224
        frames = [resize_image(frame, target_size=(target_h, target_w)) for frame in frames] 
        return frames

    def generate(self, question, video_path):
        """
        Run inference using the Video-ChatGPT model.

        Parameters:
        sample : Initial sample
        video_frames (torch.Tensor): Video frames to process.
        question (str): The question string.
        conv_mode: Conversation mode.
        model: The pretrained Video-ChatGPT model.
        vision_tower: Vision model to extract video features.
        tokenizer: Tokenizer for the model.
        image_processor: Image processor to preprocess video frames.
        video_token_len (int): The length of video tokens.

        Returns:
        dict: Dictionary containing the model's output.
        """
        video_frames = self.get_frames(video_path)
        video_frames = torch.stack([self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in video_frames])
        slice_len = video_frames.shape[0]

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        vision_tower = self.model.get_vision_tower()
        image_processor = vision_tower.image_processor

        if self.model.config.config["use_cluster"]:
            for n, m in self.model.named_modules():
                m = m.to(dtype=torch.bfloat16)

        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

        conv = conv_templates["simple"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_frames.half().cuda(),
                temperature=0,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
       