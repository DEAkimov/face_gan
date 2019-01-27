#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3
python3 src/main.py \
--architecture dc --criterion mse --loss simple \
--data_path resources/celeba --logdir logs/dc_gan/exp_1/
