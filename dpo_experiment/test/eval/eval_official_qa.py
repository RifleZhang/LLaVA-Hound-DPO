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
from data_processing.utils import format_docstring, load_json_data, save_jsonl, save_json, get_id_from_frame_path

from inference.chatgpt_inference import chatgpt_inference

# reuse chatgpt_verifier.py for now
PROMPT_TEMPLATE = """
Please evaluate the following video-based question-answer pair:

Question: {question}
Correct Answer: {answer}
Predicted Answer: {prediction}

Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. 
Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. 
For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}.
"""

SYSTEM_PROMPT_TEMPLATE = """
You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. 
Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
------
##INSTRUCTIONS: 
- Focus on the meaningful match between the predicted answer and the correct answer.
- Consider synonyms or paraphrases as valid matches.
- Evaluate the correctness of the prediction compared to the answer.
"""

RESULTING_PATH=os.environ.get("RESULTING_PATH_OFFICIAL", "./eval_official_results.jsonl")

def extract_score(input_str):
    # Regular expression to match the 'score' key and its numerical value
    # This pattern accounts for:
    # - Optional spaces around key and value
    # - Single or double quotes for the key
    # - A colon followed by optional spaces before the value
    # - Captures the numerical value associated with the 'score' key
    pattern = r"'score'\s*:\s*(\d+)|\"score\"\s*:\s*(\d+)|'score':(\d+)|\"score\":(\d+)"
    
    
    # Search for the pattern in the input string
    match = re.search(pattern, input_str)
    
    # If a match is found, return the first captured group that is not None
    if match:
        # match.groups() contains all the captured groups; filter out None and convert to int
        score = next(int(g) for g in match.groups() if g is not None)
        return score
    else:
        # Return a default value or raise an error if 'score' is not found
        return None

def maybe_truncate(text, max_len=256):
    words = text.split()
    if len(words) > max_len:
        return " ".join(words[:max_len])
    return text

def make_data(dp):
    ret = {}
    ret['id'] = dp['id']
    ret['variables'] = {
        "question": dp['question'],
        "answer": dp['answer'],
        "prediction": dp["prediction"],
    }
    return ret

def main(pred_path, output_dir, output_path, num_tasks=1, model_name='chatgpt-3.5-turbo', temperature=0, top_p=1.0, max_new_tokens=256):
    prompt_template = PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    prompt_template = format_docstring(prompt_template)
    system_prompt = format_docstring(system_prompt)
    pred_data = load_json_data(pred_path)
    
    data_to_send_list = [] 
    for sample in pred_data:
        """ pred data format
        {'id': '8467_2_0',
        'modal_path': '/mnt/bn/liangkeg/data/frames/activitynet/v_9SiYS0SEKTw-Scene-006',
        'query': "What color is the cyclist's long-sleeved shirt?",
        'anwer': 'The cyclist is wearing a white long-sleeved shirt.',
        'modal_type': 'VIDEO',
        'video_decode_backend': 'frames',
        'temperature': 0.0,
        'top_p': 0.9,
        'max_new_tokens': 1024,
        'model_prediction': {'status': 'success', 'message': 'The cyclist is wearing a white long-sleeved shirt.'}}
        """
        if sample['model_prediction']['status'] != 'success':
            logger.info(f"no valid prediction {sample['id']}")
            continue
        pred = sample['model_prediction']['message']
        pred = maybe_truncate(pred)
        
        question = sample['query']
        answer = sample['answer']
        data_to_send = {
            'id': sample['id'],
            'question': question,
            'answer': answer,
            'prediction': pred,
        }
        data_to_send = make_data(data_to_send)
        data_to_send_list.append(data_to_send)
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
                      system_prompt=system_prompt,
                      model_name=model_name,
                      parser=None,
                      **kwargs)
                      

    combined_contents = load_json_data(output_path)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for item in combined_contents:
        # Computing score
        try:
            response = item['response']
            score = extract_score(response)
            if score is not None:
                score_sum += score
                count += 1
            else:
                print(f'no score found in {response}')
            
            # Computing accuracy
            if "yes" in response.lower():
                yes_count += 1
            elif "no" in response.lower():
                no_count += 1
        except:
            print(f"invalid response: {item}")

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)
    # res = {
    #     "accuracy": accuracy,
    #     "average_score": average_score,
    #     "yes_count": yes_count,
    #     "no_count": no_count,
    # }

    resulting_dict = {
        'result': f"{accuracy*100:.2f}/{average_score:.2f}",
        'name_or_path': output_path, 
    }
    save_jsonl(RESULTING_PATH, resulting_dict, append=True)

if __name__ == "__main__":
    fire.Fire(main)

