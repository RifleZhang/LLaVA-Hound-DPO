import socket
import json
from PIL import Image
import fire
import torch 
from logzero import logger
from llava.utils import disable_torch_init

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from inference.inference_utils import ModelInference

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return image

def start_server(host, port, inference_model):
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
                result = inference_model.generate(**input_data)
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

    inference_model = ModelInference(model=model, tokenizer=tokenizer, processor=processor, context_len=context_len)
    start_server(host, port, inference_model)

if __name__ == "__main__":
    fire.Fire(main)