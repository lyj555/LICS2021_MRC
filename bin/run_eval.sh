#!/bin/bash
#
# Author: liuyongjie
# Email: 626671935@qq.com
# GitHub: https://github.com/lyj555
# Date: 2021-04-10
# Brief:
#   LICS MRC部分 evaluate
# Arguments:
#     $1: data_file,   required, raw data file path
#     $2: pred_file,   required, prediction file path
#     $3: python_path, optional, if empty, will use default python
# Example: bash run_eval.sh
# Returns:
#   succ: exit 0
#   fail: exit 1

data_file=$1
pred_file=$2
python_path=$3

# [1]. check params
[[ ! -f $data_file || ! -f $pred_file ]] && \
    echo "param data_file: $data_file and pred_file: $pred_file is not regular file!!!" && exit 1

[[ -z $python_path ]] && python_path=$(which python)
[[ ! -x $python_path ]] && echo "the python_path: $python_path is not executable!" && exit 1

sh_dir=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)  # current directory path
project_dir=${sh_dir}/..

# [2]. run evaluate(all tags)
${python_path} ${project_dir}/evaluate.py --data_file $data_file --pred_file $pred_file

# [3]. run evaluate separate tag
for tag in 'in-domain' 'vocab' 'phrase' 'semantic-role' 'fault-tolerant' 'reasoning'
do
    ${python_path} ${project_dir}/evaluate.py --data_file $data_file --pred_file $pred_file --tag $tag
done
