#!/bin/bash

# example
# ./search.sh ./experiments/xxx/xxx.yaml ./ckpt/xxx.pth min_param max_param 'vit_base_patch16_224.augreg_in21k_ft_in1k' name proxy_name data_load seed epoch

search_space=$1
af_supernet_ckpt=$2
min_pl=$3
max_pl=$4
teacher_model=$5
name=$6
measure_name=$7
data_load=$8
seed=$9
epochs=$10
gpus=1

# warmup
python -m torch.distributed.launch --nproc_per_node=$gpus --use_env evolution.py --data-path ./data/imagenet --gp \
--change_qk --relative_position --dist-eval --cfg $search_space --resume $af_supernet_ckpt \
--min-param-limits $min_pl --param-limits $max_pl --data-set EVO_IMNET \
--kendall --teacher_model $teacher_model \
--save_ckpt_path './ckpt/kendall-'$name'.pth' \
--seed $seed \
--measure-name $measure_name --data-load $data_load

#############
# Stage 1: DAZC stage
#############

# mini-sample search for n = 5

python -m torch.distributed.launch --nproc_per_node=$gpus --use_env evolution.py --data-path ./data/imagenet --gp \
--change_qk --relative_position --dist-eval --cfg $search_space \
--min-param-limits $min_pl --param-limits $max_pl --data-set EVO_IMNET \
--kendall --teacher_model $teacher_model --load_ckpt_path './ckpt/kendall-'$name'.pth' \
--cluster_json 'cls_cluster_5.json' --kendall_record_json $name'-kendall_search_5.json' \
--seed $seed --kendall_search \
--measure-name $measure_name --data-load $data_load

#############
# Stage 2: NAS stage
#############

# architecture search for n = 5

python -m torch.distributed.launch --nproc_per_node=$gpus --use_env evolution.py --data-path ./data/imagenet --gp \
--change_qk --relative_position --dist-eval --cfg $search_space --resume $af_supernet_ckpt \
--min-param-limits $min_pl --param-limits $max_pl --data-set EVO_IMNET \
--teacher_model $teacher_model \
--record_json $name'-kendall_search_5.json' \
--seed $seed \
--output './ckpt/'$name'_5' --batch-size 128 --max-epochs $epochs \
--measure-name $measure_name --data-load $data_load

