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
from inference.utils import get_score

# reuse chatgpt_verifier.py for now
PROMPT_TEMPLATE = """
Given the following inputs:

1. **Ground Truth Video Caption**: {caption}
2. **Question Related to the Caption**: {question}
3. **Ground Truth Answer**: {answer}
4. **Model Predicted Answer**: {prediction}

Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the video caption and the question. Consider the following criteria for evaluation:

- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the video caption?
- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?
- **Clarity**: Assess the clarity of the predicted answer. Look for issues such as repetition, unclear descriptions, or any grammatical errors that could hinder understanding.
- **Completeness**: Determine if the predicted answer fully covers the scope of the ground truth answer. Does it leave out critical information or does it include all necessary details?

**Output Format**:
Explanation: <brief judgement of prediction>
Score: <an integer score of quality from 1-5>
"""

RESULTING_PATH=os.environ.get("RESULTING_PATH", "./eval_results.jsonl")
def maybe_truncate(text, max_len=256):
    words = text.split()
    if len(words) > max_len:
        return " ".join(words[:max_len])
    return text

def make_data(dp):
    ret = {}
    ret['id'] = dp['id']
    ret['variables'] = {
        "caption": dp['caption'],
        "question": dp['question'],
        "answer": dp['answer'],
        "prediction": dp["prediction"],
    }
    return ret

def main(gt_path, pred_path, output_dir, output_path, num_tasks=1, model_name='chatgpt-3.5-turbo', temperature=0, top_p=1.0, max_new_tokens=256):
    prompt_template = PROMPT_TEMPLATE
    prompt_template = format_docstring(prompt_template)
    gt_data = load_json_data(gt_path)
    pred_data = load_json_data(pred_path)

    gt_caption_dict = {}
    for sample in gt_data:
        video_id = get_id_from_frame_path(sample['video'])
        gt_caption_dict[video_id] = sample['caption']

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
        video_id = get_id_from_frame_path(sample['modal_path'])
        if video_id not in gt_caption_dict:
            print(f"no gt data for {sample['id']}")
            continue
        if sample['model_prediction']['status'] != 'success':
            logger.info(f"no valid prediction {sample['id']}")
            continue
        pred = sample['model_prediction']['message']
        pred = maybe_truncate(pred)
        question = sample['query']
        answer = sample['answer']
        caption = gt_caption_dict[video_id]
        data_to_send = {
            'id': sample['id'],
            'caption': caption,
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
                      system_prompt=None,
                      model_name=model_name,
                      parser=None,
                      **kwargs)

    result = load_json_data(output_path)
    cnt = 0
    avg_score = 0
    acc = 0
    for x in result:
        try:
            score = get_score(x['response'])
            avg_score += score
            if score >=3.0:
                acc += 1
            cnt += 1
        except:
            print(f"not valid: {x}")
    logger.info(f"avg score: {avg_score/cnt:.2f}, acc: {acc/cnt*100:.2f}")
    resulting_dict = {
        'result': f"{acc/cnt*100:.2f}/{avg_score/cnt:.2f}",
        'name_or_path': output_path, 
    }
    save_jsonl(RESULTING_PATH, resulting_dict, append=True)

if __name__ == "__main__":
    fire.Fire(main)

