#!/bin/bash
export CUDA_VISIBLE_DEVICES=1; \
python3 src/main.py \
--architecture dc --criterion mse --loss simple \
--data_path resources/celeba --logdir logs/dc_gan/m_s/
