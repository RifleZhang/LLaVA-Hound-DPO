import argparse
import json
import os
import re
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def parse_pred_ans(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]
        suffix_pred_ans = pred_ans[-4:]

        if "yes" in prefix_pred_ans or "yes" in suffix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans or "no" in suffix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"

    return pred_label


def eval_live_vqa(pred_file):
    pred_res = json.load(open(pred_file))

    label_map = {
        "yes": 1,
        "no": 0,
        "other": -1,
    }

    gts, preds = [], []
    for data in pred_res:
        qas = data['qas']

        for qa in qas:
            if qa['a'] == 'yes':
                gts.append(1)
            elif qa['a'] == 'no':
                gts.append(0)
            else:
                raise ValueError

            pred = parse_pred_ans(qa['pred'].lower())
            preds.append(label_map[pred])

    print(len(gts), len(preds))
    acc = accuracy_score(gts, preds)

    # # v1
    # TP, TN, FP, FN = 0, 0, 0, 0
    # POS, NEG = 1, 0
    # other_num = 0
    # for pred, label in zip(preds, gts):
    #     if pred == -1:
    #         other_num += 1
    #         continue
    #
    #     if pred == POS and label == POS:
    #         TP += 1
    #     elif pred == POS and label == NEG:
    #         FP += 1
    #     elif pred == NEG and label == NEG:
    #         TN += 1
    #     elif pred == NEG and label == POS:
    #         FN += 1
    #
    # precision = float(TP) / float(TP + FP)
    # recall = float(TP) / float(TP + FN)
    # f1 = 2 * precision * recall / (precision + recall)
    # acc = (TP + TN) / (TP + TN + FP + FN)
    # print('Accuracy: {}, {}'.format(acc, acc1))
    # print('Precision: {}'.format(precision))
    # print('Recall: {}'.format(recall))
    # print('F1 score: {}'.format(f1))
    # print('Other Num: {}'.format(other_num))
    # print('%.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall))

    # v2
    clean_gts = []
    clean_preds = []
    other_num = 0
    for gt, pred in zip(gts, preds):
        if pred == -1:
            other_num += 1
            continue
        clean_gts.append(gt)
        clean_preds.append(pred)

    conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
    precision = precision_score(clean_gts, clean_preds, average='binary')
    recall = recall_score(clean_gts, clean_preds, average='binary')
    tp, fn = conf_mat[0]
    fp, tn = conf_mat[1]

    print(f"Acc: {round(acc, 4)}\nPrecision: {round(precision, 4)}\nRecall: {round(recall, 4)}\nF1-score: {round(2* precision * recall / (precision + recall), 4)}\nOther Num: {other_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default='')
    parser.add_argument("--output_file", type=str, default="live_vqa_res.json")
    args = parser.parse_args()

    eval_live_vqa(os.path.join(args.results_dir, args.output_file))