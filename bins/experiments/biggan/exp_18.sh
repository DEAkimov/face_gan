#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1; \
python3 src/main.py \
--architecture big --criterion mse --loss simple \
--data_path resources/celeba --logdir logs/big_gan/m_s/ \
--write_period 300 --fid_period 3000 --img_size 256 \
--n_epoch 100