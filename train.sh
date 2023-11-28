#!/bin/sh
PARTITION= Few-shot Semantic Segmentation

GPU_ID=0,1

arch=CFENet
net=resnet50 # resnet50 or resnet101
dataset=coco # pascal or coco
split_name=split0
shot_name = 1   # 1 or 5

exp_dir=exp/${arch}/${dataset}/${shot_name}shot/$split{split_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${shot_name}shot/${dataset}_$split{split_name}_${shot_name}shot_${net}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=12 train.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log