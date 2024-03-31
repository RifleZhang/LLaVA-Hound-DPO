import openai
import time
import requests
import json
from PIL import Image
from io import BytesIO
import base64


GPT4KEYPOOL = ["VrJQmRwcwnRW3KVEDaE8D9gYZm2a0zPm"]
class GPT4Wrapper:
    def __init__(self, temperature=0, model="gpt-4-32k-0613", time_out=30):
        self.key_pool = GPT4KEYPOOL
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        openai.api_base = "https://search-us.byteintl.net/gpt/openapi/online/v2/crawl"
        # openai.api_base = "https://search-us.bytedance.net/gpt/openapi/online/v2/crawl"
        openai.api_type = "azure"
        openai.api_version = "2023-06-01-preview"
        openai.api_key = self.key_pool[0]

    def request(self, query, system_content=None):
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": f"{query}"})
        response = openai.ChatCompletion.create(
            engine="gpt_openapi",
            messages=messages,
            temperature=self.temperature,
            model=self.model
        )
        return response

    def get_completion(self, user_prompt=None, system_prompt=None, max_try=10):
        gpt_cv_nlp = '[]'
        key_i = 0
        total_tokens = 0
        # gpt_cv_nlp, total_tokens = self.request(system_prompt, user_prompt)
        while max_try > 0:
            try: 
                response = self.request(system_prompt, user_prompt)
                gpt_cv_nlp = response.choices[0]['message']['content']
                total_tokens = response.usage['total_tokens']
                break
            except:
                print(f"fail {max_try}")
                key = self.key_pool[key_i % len(self.key_pool)]
                openai.api_key = key
                key_i += 1
                time.sleep(2)
                max_try -= 1

        return gpt_cv_nlp, total_tokens

def fetch_resize_and_encode_image(image_path, max_size=3):
    image = Image.open(image_path)
    image = image.convert("RGB")
    # Initialize buffer
    buffer = BytesIO()

    while True:
        # Save image to buffer
        image.save(buffer, format="JPEG")
        buffer_size = buffer.tell()

        # Check if the image size is acceptable
        if buffer_size <= max_size * 1024 * 1024:
            break
        
        # Calculate scale factor for resizing
        scale_factor = (max_size * 1024 * 1024 / buffer_size) ** 0.5
        new_dimensions = (int(image.width * scale_factor), int(image.height * scale_factor))

        # Resize the image
        image = image.resize(new_dimensions)
        
        # Reset buffer for next iteration
        buffer.seek(0)
        buffer.truncate()

    # Rewind and encode the buffer
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded_image

    
GPT4VKEYPOOL = ["fad2eab1f1fe42ff99d4ce1f45373102"]
RESOURCE_NAME = "gpt4vtest001" # Set this to the name of the Azure OpenAI resource
DEPLOYMENT_NAME = "gptvtest" # Set this to the name of the gptv model deployment
class GPT4VWrapper:
    def __init__(self, temperature=0, model=RESOURCE_NAME, max_img_size=3, max_new_tokens=512, time_out=30):
        self.key_pool = GPT4VKEYPOOL
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        self.max_img_size = max_img_size # limited to 5 (MB)
        self.max_new_tokens = max_new_tokens
        base_url = f"https://{RESOURCE_NAME}.openai.azure.com/openai/deployments/{DEPLOYMENT_NAME}"
        self.endpoint=f"{base_url}/chat/completions?api-version=2023-08-01-preview"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.key_pool[0],
        }

    def request(self, query, img_encoding, system_content=None):
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": 
                         [
                             f"{query}",
                             {"image": img_encoding}
                         ]
                        })
        data = {
            "messages": messages,
            "max_tokens": self.max_new_tokens,
        }
        # Make the API call
        response = requests.post(
            self.endpoint, 
            headers=self.headers, 
            data=json.dumps(data)
        )
        return response

    def get_completion(self, user_prompt, img_encoding=None, img_path=None, system_prompt=None, max_try=10):
        if img_encoding is None:
            if img_path is None:
                raise ValueError("img_encoding and img_path are both None")
            img_encoding = fetch_resize_and_encode_image(img_path, self.max_img_size)
        gpt_cv_nlp = '[]'
        key_i = 0
        total_tokens = 0
        # gpt_cv_nlp, total_tokens = self.request(system_prompt, user_prompt)
        while max_try > 0:
            response = self.request(query=user_prompt, img_encoding=img_encoding, system_content=system_prompt)
                # print('Succ: ', gpt_cv_nlp)
            if response.status_code == 200:
                response = response.json()
                gpt_cv_nlp = response['choices'][0]['message']['content']
                total_tokens = response['usage']['total_tokens']
                break
            else:
                print(f"fail {max_try}: {response}")
                key = self.key_pool[key_i % len(self.key_pool)]
                self.headers["api-key"] = key
                key_i += 1
                time.sleep(1)
                max_try -= 1

        return gpt_cv_nlp, total_tokens


