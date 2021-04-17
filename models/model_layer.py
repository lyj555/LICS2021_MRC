# -*- coding: utf-8 -*-

import paddle
from paddle import nn
from paddlenlp.transformers import ErniePretrainedModel, BertPretrainedModel, RobertaPretrainedModel


class ErnieForQuestionAnswering(ErniePretrainedModel):
    def __init__(self, ernie):
        super(ErnieForQuestionAnswering, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None,
                position_ids=None, attention_mask=None):
        """
        forward network
        :param input_ids: [batch_size, max_seq_length]
        :param token_type_ids: segment_ids, [batch_size, max_seq_length]
        :param position_ids: None
        :param attention_mask: None
        :return: (start_logits[batch_size, max_seq_length], end_logits, cls_logits[batch_size, 2])
        """
        # sequence_output: [batch_size, max_seq_length, hidden_size]
        # pooled_output: [batch_size, hidden_size]
        sequence_output, pooled_output = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        logits = self.classifier(sequence_output)  # 每个位置是一个二分类，[batch_size, max_seq_length, 2]
        logits = paddle.transpose(logits, perm=[2, 0, 1])  # [2, batch_size, max_seq_length]
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)  # [batch_size, max_seq_length]
        cls_logits = self.classifier_cls(pooled_output)  # [batch_size, 2]
        return start_logits, end_logits, cls_logits


class BertForQuestionAnswering(BertPretrainedModel):
    def __init__(self, bert):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = bert  # allow bert to be config
        self.classifier = nn.Linear(self.bert.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.bert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None,
                position_ids=None, attention_mask=None):
        """ refer to class `ErnieForQuestionAnswering` params note"""
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits


class RobertaForQuestionAnswering(RobertaPretrainedModel):
    def __init__(self, roberta):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.classifier_cls = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None,
                position_ids=None, attention_mask=None):
        """ refer to class `ErnieForQuestionAnswering` params note"""
        sequence_output, pooled_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits


if __name__ == "__main__":
    from paddlenlp.transformers import ErnieTokenizer
    from utils.data_helper import DataHelper

    data_path = "./dataset/dev.json"
    pretrain_model_path = "./finetuned_model"

    tokenizer = ErnieTokenizer.from_pretrained(pretrain_model_path)
    da = DataHelper(tokenizer=tokenizer,
                    batch_size=2, doc_stride=128, max_seq_length=64)
    data_loader = da.get_iterator(data_path, part_feature=False)

    a = next(iter(data_loader))

    model = ErnieForQuestionAnswering.from_pretrained(pretrain_model_path)
    input_ids, segment_ids, start_positions, end_positions, answerable_label = a
    start_logits, end_logits, cls_logits = model(input_ids=input_ids, token_type_ids=segment_ids)
