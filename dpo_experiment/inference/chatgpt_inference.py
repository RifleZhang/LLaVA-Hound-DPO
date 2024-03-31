import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
import fire
import re
from logzero import logger
from data_processing.utils import format_docstring, load_jsonl, load_json, save_jsonl, save_json

OPENAI_BACKEND=os.environ.get("OPENAI_BACKEND", 'none')
if OPENAI_BACKEND == 'azure':
    # for azure openai
    try:
        AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", None)
        API_VERSION = os.environ.get("API_VERSION", None)
        AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
        client = openai.AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION,
            api_key=AZURE_OPENAI_KEY,
        )
        print(f"setup AZURE_OPENAI client: {client}")
    except:
        raise(f"azure client not configured")
elif OPENAI_BACKEND=='openai':
    try:
        OPENAI_KEY = os.environ.get("OPENAI_KEY", None)
        ORGANIZATION_KEY = os.environ.get("ORGANIZATION_KEY", None)
        client = openai.OpenAI(
        organization=ORGANIZATION_KEY,
        api_key=OPENAI_KEY,
        )
        print(f"setup OPENAI client: {client}")
    except:
        raise(f"openai client not configured")
else:
    raise(f"openai backend not configured, choice: ['azure', 'openai']")



def openai_generate(model_name, prompt, system_prompt=None, **kwargs):
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.95)
    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })
    messages.append({
        "role": "user",
        "content": prompt
    })
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )
    output_text = chat_completion.choices[0].message.content
    return output_text

def annotate(item, prompt_template, system_prompt=None, 
             model_name='chatgpt-3.5-turbo', parser=None, **kwargs):
    idx = item['id']
    variables = item['variables']
    prompt_template = format_docstring(prompt_template)
    prompt = prompt_template.format(**variables)
    result = openai_generate(model_name=model_name, 
                             prompt=prompt,
                             system_prompt=system_prompt,
                             **kwargs)
    # result = DEBUG_MESSAGE
    result_dict = {
        'id': idx,
        'response': result,
    }
    if parser is not None:
        try:
            parsed_result = parser(result)
            result_dict['parsed_result'] = parsed_result
        except Exception as e:
            result_dict['parsed_result'] = {"error": str(e)}
    return result_dict
       
def annotate_list(data_list, prompt_template, output_dir,
                  model_name='chatgpt-3.5-turbo', system_prompt=None,  parser=None, **kwargs):
    for i, item in tqdm(enumerate(data_list)):
        try:
            result_dict = annotate(item=item, prompt_template=prompt_template, system_prompt=system_prompt,
                                   model_name=model_name,
                                   parser=parser, **kwargs)
            save_json(f"{output_dir}/{item['id']}.json", result_dict)
        except Exception as e:
            print(f"Error processing file '{item['id']}': {e}")
            message = e.message
            if 'Error code: 400' in message: # error with prompt, don't retry
                error_dict = item.copy()
                error_dict['error'] = message
                save_json(f"{output_dir}/{item['id']}.json", error_dict)
            continue

def annotate_list_wrapper(args_dict):
    """Wrapper function to unpack arguments from a dictionary and call annotate_list."""
    return annotate_list(**args_dict)
        
def gather_results(output_dir):
    combined_contents = []
    files = []
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "r") as json_file:
            content = json.load(json_file)
            combined_contents.append(content)
        files.append(file_path)
    return combined_contents, files

def clean_output_dir(files):
    for file in files:
        os.remove(file)

def chatgpt_inference(data, prompt_template, output_dir, output_path, num_tasks=1,  
                      model_name='chatgpt-3.5-turbo', system_prompt=None, parser=None, **kwargs):
    """
    Main function to control the flow of the program.
    """
    temperature = kwargs.get('temperature', 0)
    top_p = kwargs.get('top_p', 1.0)
    while True:
        try:
            contents, files = gather_results(output_dir)
            clean_output_dir(files)
            if os.path.exists(output_path):
                combined_contents = load_jsonl(output_path)
                combined_contents += contents
                result_idx = set([item['id'] for item in combined_contents])
                # missing_data = [item for i, item in enumerate(data) if i not in result_idx]
                missing_data = [item for item in data if item['id'] not in result_idx]
                data = missing_data
                logger.info(f"found resulting file, impute missing data: {len(data)}")
            else:
                combined_contents = contents

            # Break the loop when there are no incomplete files
            if len(data) == 0:
                break
            if len(data) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(data) // num_tasks
            all_parts = [data[i:i + part_len] for i in range(0, len(data), part_len)]
            task_args = [
                {
                    'data_list': data_list,
                    'prompt_template': prompt_template,
                    'output_dir': output_dir,
                    'parser': parser,
                    'system_prompt': system_prompt,
                    'model_name': model_name,
                    'temperature': temperature,
                    'top_p': top_p,
                }
                for data_list in all_parts
            ]
            with Pool() as pool:
                pool.map(annotate_list_wrapper, task_args)

            print("finish iteration")

            files = []
            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents.append(content)
                files.append(file_path)
            save_jsonl(output_path, combined_contents)
            clean_output_dir(files)
        except Exception as e:
            print(f"Error: {e}")


