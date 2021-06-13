# -*- coding: utf-8 -*


# {
#     "context": "高铁和动车上是可以充电的",
#     "qas": [
#         {
#             "question": "高铁站可以充电吗",
#             "type": "vocab_noun",
#             "id": "ebbe3fc466f0f04177b8a64d2ee0de69",
#             "answers": [
#                 {
#                     "text": "",
#                     "answer_start": -1
#                 }
#             ],
#             "is_impossible": true
#         }
#     ],
#     "title": "高铁和动车上能充电吗? - 知乎"
# }

import json


def generate_du_sample(sample):
    if not sample or not sample["documents"]:
        return None
    res = []
    question = sample["question"]
    question_id = sample['question_id']
    # 1. generate positive samples
    if sample["match_scores"] and sample['match_scores'][0] >= 0.7 and sample['answer_docs'] and \
            sample['answer_docs'][0] < len(sample['documents']) and sample['answer_spans']:
        ans_doc = sample['documents'][int(sample['answer_docs'][0])]
        split_para = ans_doc['segmented_paragraphs'][int(ans_doc['most_related_para'])]
        else_para = ''
        for i in range(len(ans_doc['segmented_paragraphs'])):
            if i != int(ans_doc['most_related_para']):
                else_para += ans_doc['paragraphs'][i]
        all_para = ''.join(split_para) + else_para
        if 50 <= len(all_para) <= 1000:
            answer_span = sample['answer_spans']
            ans_text = ''.join(split_para[answer_span[0][0]: answer_span[0][1] + 1])
            ans_start_pos = len(''.join(split_para[:answer_span[0][0]]))
            tmp_du_sample = {
                "context": all_para,
                "qas": [{
                    "question": question,
                    "type": "du_reader2.0",
                    "id": question_id,
                    "answers": [{
                        "text": ans_text,
                        "answer_start": ans_start_pos
                    }],
                    "is_impossible": False
                }],
                "title": ans_doc.get("title", "")
            }
            res.append(tmp_du_sample)
    # 2. generate negative sample
    for doc in sample["documents"]:
        if not doc["is_selected"]:
            all_para = "".join(doc["paragraphs"])
            if 50 <= len(all_para) <= 1000:
                tmp_du_sample = {
                    "context": all_para,
                    "qas": [{
                        "question": question,
                        "type": "du_reader2.0",
                        "id": question_id,
                        "answers": [{
                            "text": "",
                            "answer_start": -1
                        }],
                        "is_impossible": True
                    }],
                    "title": doc.get("title", "")
                }
                res.append(tmp_du_sample)
    return res


if __name__ == "__main__":
    data_path = ""  # du_reader2.0 data path
    rets = []
    with open(data_path) as f:
        for line in f:
            line = line.strip("\n\r")
            if not line:
                continue
            line_dic = json.loads(line)
            line_ret = generate_du_sample(line_dic)
            if not line_ret:
                continue
            rets.extend(line_ret)
