#!/bin/bash
CUDA_VISIBLE_DEVICES=1
python3 src/main.py \
--architecture dc --criterion mse --loss relativistic \
--data_path resources/celeba --logdir logs/dc_gan/exp_2/
