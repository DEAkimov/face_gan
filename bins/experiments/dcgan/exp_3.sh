#!/bin/bash
export CUDA_VISIBLE_DEVICES=3; \
python3 src/main.py \
--architecture dc --criterion bce --loss relativistic \
--data_path resources/celeba --logdir logs/dc_gan/b_r/
