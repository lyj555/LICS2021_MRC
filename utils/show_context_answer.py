# -*- coding: utf-8 -*-

import sys
import json


def show_context_answer(answer_path, raw_data_path, save_path):
    # [1]. read answer data
    with open(answer_path) as f:
        predict_answer = json.load(f)

    # [2]. read raw data
    with open(raw_data_path) as f:
        raw_data = json.load(f)

    # [3]. parse raw data
    rets = []
    for text in raw_data["data"]:  # one data contain many texts
        title = text.get("title", "").strip()  # one text only has one title
        for paragraph in text["paragraphs"]:  # one text contain many paragraphs
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question = qa["question"].strip()
                # is_impossible = qa.get("is_impossible", False)

                # answer_starts = [
                #     answer["answer_start"] for answer in qa.get("answers", [])
                # ]
                answers = [
                    answer["text"].strip() for answer in qa.get("answers", [])
                ]
                answer = "no answer" if not answers[0] else answers[0]
                if qas_id in predict_answer:
                    show_record = \
                        f"{qas_id}\t{question}\t{context}\t{answer}\t{predict_answer[qas_id]}\n"
                    rets.append(show_record)
    # save data
    with open(save_path, "w") as f:
        for line in rets:
            f.write(line)


if __name__ == "__main__":
    # answer_path = "../output/predict/0501_0945_model_name_finetuned_model/small_predictions.json"
    # raw_path = "../dataset/small_test.json"
    # save_path = "../output/predict/0501_0945_model_name_finetuned_model/small_eval.txt"
    answer_path = sys.argv[1]
    raw_path = sys.argv[2]
    save_path = sys.argv[3]
    show_context_answer(answer_path, raw_path, save_path)

