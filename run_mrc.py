# -*- coding: utf-8 -*-

import os
import logging
import json
import glob
import time
import random

import numpy as np
import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import BertTokenizer, ErnieTokenizer, RobertaTokenizer

from utils.data_helper import DataHelper
from utils.infer import compute_prediction_span
from utils.input_args import parse_args
from models.loss_layer import CrossEntropyLossForQA
from models.model_layer import ErnieForQuestionAnswering, BertForQuestionAnswering, RobertaForQuestionAnswering
from evaluate import read_mrc_dataset, read_model_prediction, evaluate, print_metrics
from utils.confirm_cls_threshold import confirm_threshold

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class ModelOperation(object):
    """ModelTrain"""

    def __init__(self):
        self.cur_process_num = paddle.distributed.get_world_size()  # PADDLE_TRAINERS_NUM 的值，默认值为1
        self.cur_process_rank = paddle.distributed.get_rank()  # PADDLE_TRAINER_ID 的值，默认值为0

        self.model_class = {
            "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
            "bert": (BertForQuestionAnswering, BertTokenizer),
            "roberta": (RobertaForQuestionAnswering, RobertaTokenizer)
        }
        self.data_helper = None

    def _initialize_run_env(self, device, seed):
        assert device in ("cpu", "gpu", "xpu"), \
            f"param device({device}) must be in ('cpu', 'gpu', 'xpu')!!!"
        paddle.set_device(device)
        if self.cur_process_num > 1:
            paddle.distributed.init_parallel_env()
        if seed:
            self.set_seed(seed)

    def _initialize_model(self, model_type, pretrained_model_path):
        assert os.path.exists(pretrained_model_path), \
            f"model path {pretrained_model_path} must exists!!!"
        logging.info(f"initialize model from {pretrained_model_path}")

        model_class, tokenizer_class = self.model_class[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_model_path)
        self.model = model_class.from_pretrained(pretrained_model_path)

        if self.cur_process_num > 1:
            self.model = paddle.DataParallel(self.model)

    def _initialize_optimizer(self, args, num_training_steps):
        self.lr_scheduler = LinearDecayWithWarmup(
            args.learning_rate, num_training_steps, args.warmup_proportion)

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in [
                p.name for n, p in self.model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ])

    def _start_train(self, args):
        # get train data loader
        train_data_loader = self.data_helper.get_iterator(args.train_data_path, shuffle=True)
        num_training_steps = args.max_train_steps if args.max_train_steps > 0 else \
            len(train_data_loader) * args.train_epochs
        logging.info("Num train examples: %d" % len(train_data_loader.dataset.data))
        logging.info("Max train steps: %d" % num_training_steps)
        # initialize optimizer
        self._initialize_optimizer(args, num_training_steps)
        # define loss function
        criterion = CrossEntropyLossForQA()

        global_step = 0
        tic_train = time.time()
        for epoch in range(args.train_epochs):
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                input_ids, segment_ids, start_positions, end_positions, answerable_label = batch

                logits = self.model(input_ids=input_ids, token_type_ids=segment_ids)
                loss = criterion(logits, (start_positions, end_positions, answerable_label))

                if global_step % args.logging_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss,
                           args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_gradients()

                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    if self.cur_process_rank == 0:
                        output_dir = \
                            os.path.join(args.output_dir, "model_{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = \
                            self.model._layers if isinstance(self.model, paddle.DataParallel) else self.model
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        print('Saving checkpoint to:', output_dir)

    @staticmethod
    def _evaluate(raw_data_path, pred_data_path, tag=None):
        ref_ans = read_mrc_dataset(raw_data_path, tag=tag)
        assert len(ref_ans) > 0, 'Find no sample with tag - {}'.format(tag)
        pred_ans = read_model_prediction(pred_data_path)
        F1, EM, ans_score, TOTAL, SKIP = evaluate(ref_ans, pred_ans, verbose=False)
        print_metrics(F1, EM, ans_score, TOTAL, SKIP, tag)

    def train_and_eval(self, args):
        self._initialize_run_env(args.device, args.seed)
        self._initialize_model(args.model_type, args.pretrained_model_path)
        self.data_helper = DataHelper(self.tokenizer, args.batch_size,
                                      args.doc_stride, args.max_seq_length)
        # start training
        if args.do_train:
            logging.info("start training...")
            self._start_train(args)
            logging.info("train success.")
        # start evaluation
        if args.do_eval:
            logging.info("start evaluating...")
            assert len(args.eval_files) == 1, "if do_eval, then eval_files must have one!!!"
            eval_file_path = args.eval_files[0]
            self.predict([eval_file_path], args.output_dir, args.max_answer_length,
                         args.cls_threshold, args.n_best_size)
            file_name = os.path.basename(eval_file_path).replace(".json", "")
            pred_file_path = os.path.join(args.output_dir, file_name + '_predictions.json')
            self._evaluate(eval_file_path, pred_file_path, args.tag)
            # confirm threshold
            confirm_threshold(eval_file_path, args.output_dir, file_name)
            logging.info("evaluate success.")
        # start predicting
        if args.do_predict:
            logging.info("start predicting...")
            self.predict(args.predict_files, args.output_dir, args.max_answer_length,
                         args.cls_threshold, args.n_best_size)
            logging.info("predict success.")

    @paddle.no_grad()
    def _predict(self, data_loader, output_dir, max_answer_length, cls_threshold,
                 n_best_size=10, prefix=""):
        self.model.eval()

        all_start_logits, all_end_logits = [], []
        all_cls_logits = []
        tic_eval = time.time()

        for batch in data_loader:
            input_ids, segment_ids = batch
            start_logits_tensor, end_logits_tensor, cls_logits_tensor = \
                self.model(input_ids, segment_ids)

            for idx in range(start_logits_tensor.shape[0]):
                if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                    print("Processing example: %d" % len(all_start_logits))
                    print('time per 1000:', time.time() - tic_eval)
                    tic_eval = time.time()

                all_start_logits.append(start_logits_tensor.numpy()[idx])
                all_end_logits.append(end_logits_tensor.numpy()[idx])
                all_cls_logits.append(cls_logits_tensor.numpy()[idx])

        all_predictions, all_nbest_json, all_cls_predictions = \
            compute_prediction_span(
                examples=data_loader.dataset.data,
                features=data_loader.dataset.new_data,
                predictions=(all_start_logits, all_end_logits, all_cls_logits),
                version_2_with_negative=True,
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
                cls_threshold=cls_threshold)

        # start save inference result
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, prefix + '_predictions.json'), "w", encoding='utf-8') as f:
            f.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

        with open(os.path.join(output_dir, prefix + '_nbest_predictions.json'), "w",
                  encoding="utf8") as f:
            f.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + u"\n")

        if all_cls_predictions:
            with open(os.path.join(output_dir, prefix + "_cls_preditions.json"), "w") as f:
                for cls_predictions in all_cls_predictions:
                    qas_id, pred_cls_label, no_answer_prob, answerable_prob = cls_predictions
                    f.write('{}\t{}\t{}\t{}\n'.format(qas_id, pred_cls_label, no_answer_prob, answerable_prob))
        self.model.train()

    def predict(self, predict_files, output_dir, max_answer_length, cls_threshold, n_best_size):
        assert predict_files is not None, "param predict_files should be set when predicting!"
        input_files = []
        for input_pattern in predict_files:
            input_files.extend(glob.glob(input_pattern))
        assert len(input_files) > 0, 'Can not find predict file in {}'.format(predict_files)
        for input_file in input_files:
            file_name = os.path.basename(input_file).replace(".json", "")
            data_loader = \
                self.data_helper.get_iterator(input_file, part_feature=True)  # no need extract position info
            self._predict(data_loader, output_dir, max_answer_length,
                          cls_threshold, n_best_size, prefix=file_name)

    @staticmethod
    def set_seed(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        paddle.seed(random_seed)


if __name__ == "__main__":
    # input_args = "--do_train 1 --train_data_path ./dataset/small.json " \
    #              "--do_eval 1 --eval_files ./dataset/small.json " \
    #              "--do_predict 0 --predict_files ./dataset/small_test.json " \
    #              "--device cpu --model_type ernie " \
    #              "--pretrained_model_path ./finetuned_model --train_epochs 1 " \
    #              "--batch_size 2 --max_seq_length 64 --max_answer_length 30"
    args = parse_args(input_arg=None)

    model_oper = ModelOperation()
    model_oper.train_and_eval(args)
