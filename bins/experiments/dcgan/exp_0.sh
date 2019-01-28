#!/bin/bash
export CUDA_VISIBLE_DEVICES=0; \
python3 src/main.py \
--architecture dc --criterion bce --loss simple \
--data_path resources/celeba --logdir logs/dc_gan/b_s/
