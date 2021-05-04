# -*- coding: utf-8 -*-

import os
import numpy as np

from evaluate import read_mrc_dataset, read_model_prediction, evaluate


def read_cls_prediction(cls_path):
    ans_ret = {}
    with open(cls_path) as f:
        for line in f:
            q_id, _, _, ans_prob = line.strip().split("\t")
            if q_id in ans_ret:
                ans_ret[q_id] = max(ans_ret[q_id], float(ans_prob))
            else:
                ans_ret[q_id] = float(ans_prob)
    return ans_ret


def _pred_ans_by_thresh(pred_dir, pred_file_prefix, candi_thresh):
    predictions_path = os.path.join(pred_dir, f"{pred_file_prefix}_predictions.json")
    cls_path = os.path.join(pred_dir, f"{pred_file_prefix}_cls_preditions.json")
    pred_text = read_model_prediction(predictions_path)
    pred_prob = read_cls_prediction(cls_path)
    for q_id, q_prob in pred_prob.items():
        if q_prob < candi_thresh:
            pred_text[q_id] = "no answer"
    return pred_text


def confirm_threshold(raw_data_file_path, pred_dir, pred_file_prefix):
    ref_ans = read_mrc_dataset(raw_data_file_path, tag=None)
    candidate_thresholds = np.linspace(0, 1, 100)
    ret_metrics = []
    for ind, candi_thresh in enumerate(candidate_thresholds):
        pred_ans = _pred_ans_by_thresh(pred_dir, pred_file_prefix, candi_thresh)
        F1, EM, _, _, _ = evaluate(ref_ans, pred_ans)
        ret_metrics.append((candi_thresh, F1, EM))
        if (ind + 1) % 20 == 0:
            print(f"now {ind + 1}/{len(candidate_thresholds)}, F1 is {F1}, EM is {EM}")
    ret_metrics = sorted(ret_metrics, key=lambda x: (x[1], x[2]))
    print("the best metrics&threshold is ", ret_metrics[-1])
    return ret_metrics[-1]


if __name__ == "__main__":
    raw_data_file_path = "../dataset/dev.json"
    pred_dir = "./output/eval/0411_1418"
    pred_file_prefix = "dev"
    confirm_threshold(raw_data_file_path, pred_dir, pred_file_prefix)
