#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python3 src/main.py \
--architecture sa --criterion bce --loss simple \
--data_path resources/celeba --logdir logs/sa_gan/exp_6/ \
--batch_size 2 --img_size 128
