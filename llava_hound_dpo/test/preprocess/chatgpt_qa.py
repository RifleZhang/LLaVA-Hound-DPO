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
from data_processing.utils import format_docstring, load_json_data, save_jsonl, save_json
from inference.chatgpt_inference import chatgpt_inference

PROMPT_TEMPLATE = """
Task Instructions:

Given a caption that summarizes the content of a video, generate three question-answer pairs that relate directly to the information and context provided in the caption. The questions should be grounded to the understanding of the video content.

Guidelines for QA Generation:

1. Helpfulness: Answers should provide sufficient detail and depth to fully address the question. They should include relevant explanations, or context where appropriate, to enhance understanding.

2. Faithfulness: The answers must accurately reflect the information presented in the video caption. Avoid speculation or the inclusion of information not contained or implied by the caption to maintain the integrity of the content.

3. Diversity: Craft questions that cover different aspects of the video caption to provide a comprehensive understanding of the content. This includes factual inquiries, inferential questions, and those that may elicit explanatory responses.

Input Video Caption:
{caption}

Output format:
Q1: <question1>
A1: <answer1>
Q2: <question2>
A2: <answer2>
Q3: <question3>
A3: <answer3>
"""

SYSTEM_PROMPT_TEMPLATE = """
"""

def parse_output(output):
    pattern = r"Q\d+: (.*?)\nA\d+: (.*?)\n*(?=\nQ\d+:|\n*$)"
    # Find all matches of the pattern in the output
    matches = re.findall(pattern, output, re.DOTALL)
    
    qas = []
    for question, answer in matches:
        qas.append(
            {
                "human": question,
                "gpt": answer,
            }
        )
    return qas

def main(data_path, output_dir, output_path, num_tasks=1, num_samples=None,
         model_name='chatgpt-3.5-turbo', temperature=0, top_p=1.0, max_new_tokens=256):
    prompt_template = PROMPT_TEMPLATE
    prompt_template = format_docstring(prompt_template)
    data = load_json_data(data_path)
    if num_samples is not None:
        data = data[:num_samples]
    data_to_send_list = []
    for item in data:
        item['variables'] = {
            "caption": item['conversations'][-1]['value'].strip(),
        }
        data_to_send_list.append(item)
  
    ids = [x['id'] for x in data_to_send_list]
    print(f"num ids: {len(set(ids))}")
    print(data_to_send_list[0])
    kwargs = {
        'temperature': temperature,
        "top_p": top_p,
    }

    # import pdb; pdb.set_trace()

    chatgpt_inference(data=data_to_send_list,
                    prompt_template=prompt_template,
                    output_dir=output_dir,
                    output_path=output_path,
                    num_tasks=num_tasks,
                    system_prompt=None,
                    model_name=model_name,
                    parser=None,
                    **kwargs)

if __name__ == "__main__":
    fire.Fire(main)