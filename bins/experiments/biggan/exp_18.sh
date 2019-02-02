#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3; \
python3 src/main.py \
--architecture big --criterion mse --loss simple \
--data_path resources/celeba --logdir logs/big_gan/m_s/ \
--n_discriminator 1 \
--write_period 100 --fid_period 300 --batch_size 8 \
--img_size 256 --orthogonal_penalty 0 --n_epoch 10
