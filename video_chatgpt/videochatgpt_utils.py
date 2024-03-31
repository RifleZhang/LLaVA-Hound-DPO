import os
import numpy as np
from PIL import Image
import dataclasses
from enum import auto, Enum
from typing import List
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from transformers import StoppingCriteria
from videochatgpt.video_chatgpt import VideoChatGPTLlamaForCausalLM
import torch


CACHE_DIR=os.environ.get("CACHE_DIR", None)

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }



conv_video_chatgpt_v1 = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    # system="",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

# Defining model
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

MODAL_TOKEN_LIST = [DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN]
def remove_special_tokens(text):
    for token in MODAL_TOKEN_LIST:
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

def initialize_model(model_name, projection_path=None, device_map={"": 0}):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """
    kwargs = {"device_map": device_map,
              "cache_dir": CACHE_DIR,
            #   "attn_implementation": 'flash_attention_2', 
    }

    # Disable initial torch operations
    disable_torch_init()

    # Convert model name to user path
    model_name = os.path.expanduser(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                         use_cache=True, **kwargs)

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16, **kwargs)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    # model = model.cuda()

    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # Set video token length
    video_token_len = 356

    return model, vision_tower, tokenizer, image_processor, video_token_len


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

class ModelInference:
    def __init__(self, model, tokenizer, image_processor, vision_tower, video_token_len):
        self.model=model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_tower = vision_tower
        self.video_token_len = video_token_len

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

    def get_spatio_temporal_features_torch(self, features):
        t, s, c = features.shape
        temporal_tokens = torch.mean(features, dim=1)
        padding_size = 100 - t
        if padding_size > 0:
            temporal_tokens = torch.cat((temporal_tokens, torch.zeros(padding_size, c, device=features.device)), dim=0)

        spatial_tokens = torch.mean(features, dim=0)
        concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

        return concat_tokens

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
        video_token_len = self.video_token_len

        # Prepare question string for the model
        if self.model.get_model().vision_config.use_vid_start_end:
            qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
        else:
            qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

        # Prepare conversation prompt
        conv = conv_video_chatgpt_v1.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize the prompt
        inputs = self.tokenizer([prompt])

        # Preprocess video frames and get image tensor
        image_tensor = self.image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

        # Move image tensor to GPU and reduce precision to half
        image_tensor = image_tensor.half().cuda()

        # Generate video spatio-temporal features
        with torch.no_grad():
            image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
        video_spatio_temporal_features = self.get_spatio_temporal_features_torch(frame_features)

        # Move inputs to GPU
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # Define stopping criteria for generation
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # Run model inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                stopping_criteria=[stopping_criteria]
        )

        # Check if output is the same as input
        n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        # Decode output tokens
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Clean output string
        outputs = outputs.strip().rstrip(stop_str).strip()

        return outputs