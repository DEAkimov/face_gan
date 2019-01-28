#!/bin/bash
export CUDA_VISIBLE_DEVICES=2; \
python3 src/main.py \
--architecture dc --criterion hinge --loss simple \
--data_path resources/celeba --logdir logs/dc_gan/h_s/
