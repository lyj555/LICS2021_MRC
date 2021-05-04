#!/bin/bash
#
# Author: liuyongjie
# Email: 626671935@qq.com
# GitHub: https://github.com/lyj555
# Date: 2021-04-10
# Brief:
#   LICS MRC部分 对验证集进行 预测&评估
# Arguments:
#     $1: model_type, required, ernie or bert or roberta
#     $2: model_path, required, the pretrained model path
#     $3: device, optional, cpu or gpu, default is gpu
#     $4: dev_data_path, optional, default is ./dataset/dev.json
#     $5: cls_threshold, optional, default is 0.7
#     $6: python_path, optional, if empty, will use default python
# Example: bash run_train.sh
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

[[ -z $model_type ]] && echo "param model_type is empty!!!" && exit 1
[[ ! -d $model_path ]] && echo "model_path: $model_path is not valid directory" && exit 1
[[ -z $device ]] && device="gpu"
[[ -z $dev_data_path ]] && dev_data_path=${project_dir}/dataset/dev.json
[[ -z $cls_threshold ]] && cls_threshold=0.7
[[ -z $python_path ]] && python_path=$(which python)
[[ ! -x $python_path ]] && echo "the python_path: $python_path is not executable!" && exit 1

output_dir=${project_dir}/output/eval/$(date +%m%d_%H%M)  # output directory

batch_size=2
max_seq_length=512
max_answer_length=512
doc_stride=128

${python_path} ${project_dir}/run_mrc.py \
    --do_eval 1 --eval_files $dev_data_path \
    --do_predict 0 --do_train 0 \
    --output_dir $output_dir \
    --device $device --seed 666666 --model_type $model_type \
    --pretrained_model_path $model_path \
    --batch_size $batch_size \
    --max_seq_length $max_seq_length \
    --max_answer_length $max_answer_length --doc_stride $doc_stride \
    --cls_threshold $cls_threshold

# [2]. show real & predict answer
base_name=$(basename $dev_data_path)
file_name=${base_name%.json}
answer_path=$output_dir/${file_name}_predictions.json
raw_path=$dev_data_path
save_path=$output_dir/${file_name}_answers.txt

# [3]. concat text and answer for debugging
${python_path} ${project_dir}/utils/show_context_answer.py $answer_path $raw_path $save_path
