import os
import json
import argparse

def eval_live(label_file):
    # label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    tests = json.load(open('/mnt/bd/bohanzhaiv0/MASP_PROD/benchmark_data/obj_and_policy_eval_v0.json'))
    tests_objkeys = set([test['object_id'] for test in tests])
    print(tests)
    pred_file = json.load(open(label_file))
    pred_list = []
    label_list = []
    for pred in pred_file:
        if not pred['object_id'] in tests_objkeys:
            continue
        for yes_ans in pred['pred_yes']:
            pred_list.append(yes_ans)
            label_list.append('yes')
        for no_ans in pred['pred_no']:
            pred_list.append(no_ans)
            label_list.append('no')


    new_pred_list = []
    for answer, label in zip(pred_list, label_list):
        print(answer)
        # Only keep the first sentence
        if 'No' in answer or 'not' in answer or 'no' in answer:
            new_pred_list.append(0)
        else:
            new_pred_list.append(1)
        

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    # pred_list = []
    # for answer in answers:
    #     if answer['text'] == 'no':
    #         pred_list.append(0)
    #     else:
    #         pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = new_pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(new_pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, default="/mnt/bd/bohanzhaiv0/MASP_PROD/ouputs/live_vqa.json")
    args = parser.parse_args()


    eval_live(args.result_file)
    # for file in os.listdir(args.annotation_dir):
    #     assert file.startswith('coco_pope_')
    #     assert file.endswith('.json')
    #     category = file[10:-5]
    #     cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
    #     print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
    #     eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
    #     print("====================================")
