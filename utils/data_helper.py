# -*- coding: utf-8 -*-

import json

import paddle
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.data import Pad, Stack, Dict


class DataReader(DatasetBuilder):
    """DataReader"""
    @staticmethod
    def read_raw_data(file_path):
        """ read raw input data """
        with open(file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        return input_data

    def _read(self, filename, *args):
        """ rewrite parent class's method """
        input_data = self.read_raw_data(filename)
        for text in input_data["data"]:  # one data contain many texts
            title = text.get("title", "").strip()  # one text only has one title
            for paragraph in text["paragraphs"]:   # one text contain many paragraphs
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    is_impossible = qa.get("is_impossible", False)

                    answer_starts = [
                        answer["answer_start"] for answer in qa.get("answers", [])
                    ]
                    answers = [
                        answer["text"].strip() for answer in qa.get("answers", [])
                    ]

                    yield {
                        'id': qas_id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts,
                        'is_impossible': is_impossible
                    }

    def _get_data(self, mode: str):
        pass


class DataHelper(object):
    def __init__(self, tokenizer, batch_size, doc_stride, max_seq_length):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.doc_stride = doc_stride
        self.max_seq_length = max_seq_length
        self._train_input_fn = Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "start_positions": Stack(dtype="int64"),
            "end_positions": Stack(dtype="int64"),
            "answerable_label": Stack(dtype="int64")
            })
        self._dev_input_fn = Dict({
                "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
                "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
            })

    def get_iterator(self, input_file_path, shuffle=False, part_feature=False):
        """
        get data iterator
        :param input_file_path: str, input file path
        :param shuffle: bool, if or not shuffle data
        :param part_feature: bool, if or not extract position's feature(part of feature)
        :return: paddle.io.DataLoader
        """
        data_reader = DataReader().read(input_file_path)
        data_reader.map(lambda x: self.extract_features(x, part=part_feature), batched=True)
        if part_feature:
            batch_sampler = paddle.io.BatchSampler(data_reader,
                                                   batch_size=self.batch_size, shuffle=shuffle)
            data_batch_fn = lambda samples, fn=self._dev_input_fn: fn(samples)
        else:
            batch_sampler = paddle.io.DistributedBatchSampler(
                data_reader, batch_size=self.batch_size, shuffle=shuffle)
            data_batch_fn = lambda samples, fn=self._train_input_fn: fn(samples)

        return paddle.io.DataLoader(
            dataset=data_reader,
            batch_sampler=batch_sampler,
            collate_fn=data_batch_fn,
            return_list=True)

    def extract_features(self, examples, part=False):
        """
        extract features from input data so as to feed into model
        :param examples: list[dict], the DataReader's return
        :param part: bool, if True, only extract two ids(input id and segment ids)
            if False, two ids(segment ids) + three label(start_position, end_position, answerable label)
        :return: list[dict], if part=Ture, length=2, otherwise, length=5
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            stride=self.doc_stride,
            max_seq_len=self.max_seq_length)

        for token_example in tokenized_examples:
            # Grab the sequence corresponding to that
            # example (to know what is the context and what is the question).
            sequence_ids = token_example['token_type_ids']

            # One example can give several spans,
            # this is the index of the example containing this span of text.
            sample_index = token_example['overflow_to_sample']
            if part:
                token_example["example_id"] = examples[sample_index]['id']

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                token_example["offset_mapping"] = [
                    (o if sequence_ids[k] == 1 else None)
                    for k, o in enumerate(token_example["offset_mapping"])
                ]
            else:
                # We will label impossible answers with the index of the CLS token.
                input_ids = token_example["input_ids"]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # The offset mappings will give us a map from token to character position
                # in the original context. This will help us compute the start_positions and end_positions.
                offsets = token_example["offset_mapping"]

                # One example can give several spans, this is the index of the example containing this span of text.
                answers = examples[sample_index]["answers"]
                answer_starts = examples[sample_index]["answer_starts"]

                # If no answers are given, set the cls_index as answer.
                if len(answer_starts) == 0:
                    token_example["start_positions"] = cls_index
                    token_example["end_positions"] = cls_index
                    token_example["answerable_label"] = 0
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answer_starts[0]
                    end_char = start_char + len(answers[0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 2
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    # Detect if the answer is out of the span
                    # (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and
                            offsets[token_end_index][1] >= end_char):
                        token_example["start_positions"] = cls_index
                        token_example["end_positions"] = cls_index
                        token_example["answerable_label"] = 0
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and \
                                offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        token_example["start_positions"] = token_start_index - 1
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        token_example["end_positions"] = token_end_index + 1
                        token_example["answerable_label"] = 1
        return tokenized_examples


if __name__ == "__main__":
    from paddlenlp.transformers import ErnieTokenizer

    data_path = "./dataset/dev.json"
    pretrain_model_path = "./finetuned_model"

    tokenizer = ErnieTokenizer.from_pretrained(pretrain_model_path)
    a = tokenizer(["今天天气不错", "酿豆腐按到法"], ["如何进行", "你在吗"], stride=5, max_seq_len=10)

    da = DataHelper(tokenizer=tokenizer,
                    batch_size=2, doc_stride=128, max_seq_length=512)
    data_loader = da.get_iterator(data_path, part_feature=False)
    samples_label = next(iter(data_loader))
    print(len(samples_label))

    data_loader = da.get_iterator(data_path, part_feature=True)
    samples_no_label = next(iter(data_loader))
    print(len(samples_no_label))
