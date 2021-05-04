#!/bin/bash
#
# Author: liuyongjie
# Email: 626671935@qq.com
# GitHub: https://github.com/lyj555
# Date: 2021-04-10
# Brief:
#   LICS MRC部分模型训练
# Arguments:
#     $1: model_type, required, ernie or bert or roberta
#     $2: train_data_path, optional, train data path
#     $3: dev_data_path, optional, dev data path
#     $4: pretrained_model_path, required, the pretrained model path
#     $5: device, optional, cpu or gpu, default is gpu
#     $6: python_path, optional, if empty, will use default python
# Example: bash run_train.sh ernie <pretrained path>
# Returns:
#   succ: exit 0
#   fail: exit 1

model_type=$1
train_data_path=$2
dev_data_path=$3
pretrained_model_path=$4
device=$5
python_path=$6

[[ -z $model_type ]] && echo "param model_type can't empty, should in (ernie, bert, roberta)!!!" && exit 1
[[ ! -d $pretrained_model_path ]] && echo "param pretrained_model_path must be directory!!!" && exit 1
[[ -z $device ]] && device="cpu"
[[ -z $python_path ]] && python_path=$(which python)
[[ ! -x $python_path ]] && echo "the python_path: $python_path is not executable!" && exit 1

sh_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)  # current directory path
project_dir=${sh_dir}/..

# model_type="ernie"

[[ -z $train_data_path ]] && train_data_path=${project_dir}/dataset/small.json
[[ -z $dev_data_path ]] && dev_data_path=${project_dir}/dataset/small.json
output_dir=${project_dir}/output/train/$(date +%m%d_%H%M)  # output directory

train_epochs=1
batch_size=2
max_seq_length=512
max_answer_length=512
doc_stride=128
# optimizer(adamX)
learning_rate=3e-5
weight_decay=0.01
adam_epsilon=1e-8
warmup_proportion=0.1
# steps
logging_steps=500
save_steps=100000


${python_path} ${project_dir}/run_mrc.py \
    --do_train 1 --train_data_path $train_data_path \
    --do_eval 1 --eval_files $dev_data_path \
    --do_predict 0 \
    --output_dir $output_dir \
    --device $device --seed 666666 --model_type $model_type \
    --pretrained_model_path $pretrained_model_path \
    --train_epochs $train_epochs --batch_size $batch_size \
    --max_seq_length $max_seq_length \
    --max_answer_length $max_answer_length --doc_stride $doc_stride \
    --logging_steps $logging_steps --save_steps $save_steps \
    --learning_rate $learning_rate --weight_decay $weight_decay \
    --warmup_proportion $warmup_proportion --adam_epsilon $adam_epsilon
