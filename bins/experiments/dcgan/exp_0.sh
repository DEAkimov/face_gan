#!/bin/bash
CUDA_VISIBLE_DEVICES=3
python3 src/main.py \
--architecture dc --criterion bce --loss simple \
--data_path resources/celeba --logdir logs/dc_gan/exp_0/
