# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# just for debug

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

set -x
unset CUDA_VISIBLE_DEVICES

task_name="llama_auto_dp2mp2pp2"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"
export GLOG_v=0
export PYTHONPATH=../../../:$PYTHONPATH
# export FLAGS_cudnn_deterministic=1
# export FLAGS_embedding_deterministic=1
# export FLAGS_enable_auto_parallel_align_mode=1
export FLAGS_enable_sharding_stage1_tensor_fusion=1

master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36677
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID
nnodes=4

source /root/paddlejob/workspace/env_run/zzh/venv/bin/activate

# python -u  -m paddle.distributed.launch \
#     --log_dir "basic_log" \
#     run_pretrain_auto.py pretrain-baichuan_13b.json

python -u  -m paddle.distributed.launch \
    --log_dir "basic_log" \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    run_pretrain_auto.py pretrain-baichuan_13b.json
