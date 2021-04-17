# -*- coding: utf-8 -*-

import argparse


def parse_args(input_arg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # input & output path config
    parser.add_argument("--do_train", default=1, type=int, help="Whether to train the model.")
    parser.add_argument("--train_data_path", type=str, default=None, help="Train data path.")
    parser.add_argument("--do_eval", default=1, type=int, help="Whether or not do evaluation")
    parser.add_argument("--eval_files", type=str, default=None, nargs='+',
                        help="evaluation data path, can have many files.")
    parser.add_argument("--do_predict", default=1, type=int, help="Whether or not predict")
    parser.add_argument("--predict_files", type=str, default=None, nargs='+',
                        help="predict data path, can have many files.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    # run environment config
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--device", type=str, default="gpu", help="Device for selecting for the training.")

    # model config
    parser.add_argument("--model_type", default="ernie", type=str,
                        help="Type of pre-trained model, now supported values is [ernie, bert, roberta] ")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="the pre-trained model path")
    # model train
    parser.add_argument("--train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    ## model optimization
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Proportion of training steps to perform linear learning rate warmup for.")
    # model steps
    parser.add_argument("--max_train_steps", default=-1,  type=int,
                        help="If > 0: set total number of training steps to perform. Override train_epochs.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    # data load
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_answer_length", type=int, default=30, help="Max answer length.")
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    # predict
    parser.add_argument("--cls_threshold", type=float, default=0.5, help="No answer threshold")
    parser.add_argument("--n_best_size", type=int, default=10,
                        help="The total number of n-best predictions to "
                             "generate in the nbest_predictions.json output file.")
    parser.add_argument('--tag', default=None, help="sample type used for evaluation")

    if input_arg:
        args = parser.parse_args(input_arg.split())
    else:
        args = parser.parse_args()
    return args
