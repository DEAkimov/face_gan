#!/bin/bash
export CUDA_VISIBLE_DEVICES=1; \
python3 src/main.py \
--architecture sa --criterion mse --loss simple \
--data_path resources/celeba --logdir logs/sa_gan/m_s/ \
--write_period 30 --fid_period = 300 --img_size 128 \
--n_epoch 10