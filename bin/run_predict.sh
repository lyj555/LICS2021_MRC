#!/bin/bash
#
# Author: liuyongjie
# Email: 626671935@qq.com
# GitHub: https://github.com/lyj555
# Date: 2021-04-10
# Brief:
#   LICS MRC部分 predict
# Arguments:
#     $1: model_type, required, ernie or bert or roberta
#     $2: model_path, required, the pretrained model path
#     $3: device, optional, cpu or gpu, default is gpu
#     $4: predict_data_path, optional, default is ./dataset/test.json
#     $5: cls_threshold, optional, default is 0.7
#     $6: python_path, optional, if empty, will use default python
# Example: bash run_eval.sh
# Returns:
#   succ: exit 0
#   fail: exit 1

model_type=$1
model_path=$2
device=$3
dev_data_path=$4
cls_threshold=$5
python_path=$6

sh_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)  # current directory path
project_dir=${sh_dir}/..

# [1]. check params
[[ -z $model_type ]] && echo "param model_type is empty!!!" && exit 1
[[ ! -d $model_path ]] && echo "model_path: $model_path is not valid directory" && exit 1
[[ -z $device ]] && device="gpu"
[[ -z $predict_data_path ]] && predict_data_path=${project_dir}/dataset/test.json
[[ -z $cls_threshold ]] && cls_threshold=0.7
[[ -z $python_path ]] && python_path=$(which python)
[[ ! -x $python_path ]] && echo "the python_path: $python_path is not executable!" && exit 1

pretrained_model_path=$model_path
model_name=$(basename $pretrained_model_path)
output_dir=${project_dir}/output/predict/$(date +%m%d_%H%M)_model_name_$model_name  # output directory

batch_size=2
max_seq_length=512
max_answer_length=512
doc_stride=128

n_best_size=10

# [2]. do predict
${python_path} ${project_dir}/run_mrc.py \
    --do_train 0 --do_eval 0 \
    --do_predict 1 --predict_files $predict_data_path \
    --output_dir $output_dir \
    --device $device --model_type $model_type \
    --pretrained_model_path $pretrained_model_path \
    --batch_size $batch_size \
    --max_seq_length $max_seq_length \
    --max_answer_length $max_answer_length --doc_stride $doc_stride \
    --cls_threshold=$cls_threshold --n_best_size=$n_best_size
