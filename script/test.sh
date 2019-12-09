#!/bin/bash

gpu=0
arch=vgg16_gap
name=test22
batch=32
# You must change
data_root="/srv/PascalVOC/VOCdevkit/VOC2012/"
test_list="datalist/PascalVOC/test.txt"

CUDA_VISIBLE_DEVICES=${gpu} python test.py \
    --arch ${arch} \
    --name ${name} \
    --data ${data_root} \
    --batch-size ${batch} \
    --test-list ${test_list} \

