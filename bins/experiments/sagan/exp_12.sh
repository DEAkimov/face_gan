#!/bin/bash
export CUDA_VISIBLE_DEVICES=3; \
python3 src/main.py \
--architecture sa --criterion bce --loss relativistic \
--data_path resources/celeba --logdir logs/sa_gan/b_r/ \
--write_period 30 --fid_period 300 --img_size 128 \
--n_epoch 10