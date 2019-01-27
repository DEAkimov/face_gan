#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python3 src/main.py \
--architecture dc --criterion bce --loss relativistic \
--data_path resources/celeba --logdir logs/dc_gan/exp_2/
