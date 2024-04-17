import os
import numpy as np
from PIL import Image
import dataclasses
from enum import auto, Enum
from typing import List
import torch

from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX

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

MODAL_TOKEN_LIST=["<video>", "<image>"]

def remove_special_tokens(text):
    for token in MODAL_TOKEN_LIST:
        if token in text:
            text = text.replace(token, "").strip()
    return text

def decode2frame(video_path, frame_dir=None, verbose=False):
    if frame_dir is None:
        frame_dir, _ = os.path.splitext(video_path)
    os.makedirs(frame_dir, exist_ok=True)
    output_dir = f"{frame_dir}/c01_%04d.jpeg"
    cmd = 'ffmpeg -loglevel quiet -i {} -vf "scale=336:-1,fps=2" {}'.format(video_path, output_dir)
    if verbose:
        print(cmd)
    os.system(cmd)

class ModelInference:
    def __init__(self, model, tokenizer, processor, context_len):
        self.model=model
        self.tokenizer = tokenizer
        self.processor = processor
        self.context_len = context_len

    @torch.no_grad()
    def generate(self, question, modal_path, modal_type='VIDEO', video_decode_backend='frames', temperature=0.7, top_p=0.9, max_new_tokens=512, **kwargs):
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
        tokenizer = self.tokenizer
        model = self.model
        processor = self.processor
        video_processor = processor.get('video', None)
        image_processor = processor.get('image', None)

        qs = remove_special_tokens(question)
        if modal_type != "TEXT":
            if model.config.mm_use_x_start_end:
                qs = DEFAULT_X_START_TOKEN[modal_type] + DEFAULT_X_TOKEN[modal_type] + DEFAULT_X_END_TOKEN[modal_type] + '\n' + qs
            else:
                qs = DEFAULT_X_TOKEN[modal_type] + '\n' + qs

        conv_mode = "v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        if modal_type == 'IMAGE':
            modal_tensor = image_processor.preprocess(modal_path, return_tensors='pt')['pixel_values'][0].half().to('cuda')
        elif modal_type == 'VIDEO':
            modal_tensor = video_processor(modal_path, return_tensors='pt', video_decode_backend=video_decode_backend)['pixel_values'][0].half().to('cuda')
        elif modal_type == 'TEXT':
            modal_type='IMAGE' # placeholder
            modal_tensor = torch.zeros(3, 224, 224).half().to('cuda') # placeholder, not processed anyway, no special token in prompt
        else:
            raise ValueError(f"modal_type {modal_type} not supported")

        # print(video_tensor.shape)
        input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[modal_type], return_tensors='pt').unsqueeze(0).to('cuda')

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if temperature < 0.01:
            do_sample=False
        else:
            do_sample=True
        max_context_length = getattr(
            model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(max_context_length - input_ids.shape[1], max_new_tokens)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[[modal_tensor], [modal_type.lower()]],
                do_sample=do_sample,
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
        return outputs