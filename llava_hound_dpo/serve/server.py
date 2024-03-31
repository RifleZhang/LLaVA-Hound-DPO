import socket
import json
from PIL import Image
import fire
import torch 
from logzero import logger
from llava.utils import disable_torch_init

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_X_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

# video_formats = ['.mp4', '.avi', '.mov', '.mkv']

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return image

def model_function(model_dict, input_data):
    # unpack model dict
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    processor = model_dict["processor"]
    video_processor = processor.get('video', None)
    image_processor = processor.get('image', None)
    context_len = model_dict["context_len"]
    modal_type = input_data.get('modal_type', 'VIDEO').upper()

    # data

    qs = input_data['query']
    if modal_type != "TEXT":
        if model.config.mm_use_x_start_end:
            qs = DEFAULT_X_START_TOKEN[modal_type] + DEFAULT_X_TOKEN[modal_type] + DEFAULT_X_END_TOKEN[modal_type] + '\n' + qs
        else:
            qs = DEFAULT_X_TOKEN[modal_type] + '\n' + qs
    print(qs)

    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if modal_type == 'IMAGE':
        image_path = input_data['image_path']
        modal_tensor = image_processor.preprocess(image_path, return_tensors='pt')['pixel_values'][0].half().to('cuda')
    elif modal_type == 'VIDEO':
        video_decode_backend = input_data.get('video_decode_backend', 'decord')
        video_path = input_data['video_path']
        modal_tensor = video_processor(video_path, return_tensors='pt', video_decode_backend=video_decode_backend)['pixel_values'][0].half().to('cuda')
    elif modal_type == 'TEXT':
        modal_type='IMAGE' # placeholder
        modal_tensor = torch.zeros(3, 224, 224).half().to('cuda') # placeholder
    else:
        raise ValueError(f"modal_type {modal_type} not supported")

    # print(video_tensor.shape)
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[modal_type], return_tensors='pt').unsqueeze(0).to('cuda')
    print(input_ids)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = input_data.get('temperature', 0.7)
    if temperature < 0.01:
        temperature = -1 # greedy
    top_p = input_data.get('top_p', 0.9)
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = input_data.get("max_new_tokens", 1024)
    max_new_tokens = min(max_context_length - input_ids.shape[1], max_new_tokens)
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[[modal_tensor], [modal_type.lower()]],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            # stopping_criteria=[stopping_criteria]
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
    result = {
        'status': 'success',
        'message': outputs
    }
    # print(outputs)
    return result

def start_server(host, port, model_dict):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server started on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        with client_socket:
            data = b''
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                data += chunk
                if b'__end__' in data:
                    # Remove the end marker
                    data = data.replace(b'__end__', b'')
                    break

            input_data = json.loads(data.decode('utf-8'))
            try:
                result = model_function(model_dict, input_data)
                client_socket.sendall(json.dumps(result).encode('utf-8'))
            except Exception as e:
                result = {
                    'status': 'error',
                    'message': str(e)
                }
                client_socket.sendall(json.dumps(result).encode('utf-8'))

def main(host='127.0.0.1', port=7767, model_path="LanguageBind/Video-LLaVA-7B", model_base=None, legacy=False):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    logger.info(f"model {model_name}")
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base = model_base, model_name=model_name, legacy=legacy)
    model = model.to('cuda')
    # import pdb; pdb.set_trace()

    model_dict = {
        "tokenizer": tokenizer,
        "model": model,
        "processor": processor,
        "context_len": context_len
    }
    start_server(host, port, model_dict)

if __name__ == "__main__":
    fire.Fire(main)