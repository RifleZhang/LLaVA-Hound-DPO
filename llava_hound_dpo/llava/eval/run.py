import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from argparse import ArgumentParser

def main(args):
    disable_torch_init()
    # video = '/mnt/bn/liangkeg/ruohongz/vllm/video_llava/llava/serve/examples/sample_demo_1.mp4'
    video= '/mnt/bn/algo-masp-nas-2/baiyi.by/data/masp_videos/7277823767945003522/v09044g40000cjnl1brc77uc0p5sestg'

    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = args.cache_dir
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    video_processor = processor['video']

    video_tensor = video_processor(video, return_tensors='pt', video_decode_backend='frames')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    key = ['video']

    inp = 'Why is this video funny?'
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[tensor, key],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_dir', help='Directory containing video files.', type=str, default="")
    parser.add_argument('--validation_data', type=str,
                        default="/mnt/bn/algo-masp-nas-2/baiyi.by/data/Benchmarks/LiveVQA/meta_data.json")
    parser.add_argument('--cache_dir', type=str, default='./')

    args = parser.parse_args()
    main(args)