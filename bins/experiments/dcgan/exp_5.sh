#!/bin/bash
export CUDA_VISIBLE_DEVICES=1; \
python3 src/main.py \
--architecture dc --criterion hinge --loss relativistic \
--data_path resources/celeba --logdir logs/dc_gan/h_r/
