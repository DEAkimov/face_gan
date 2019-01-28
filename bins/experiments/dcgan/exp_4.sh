#!/bin/bash
export CUDA_VISIBLE_DEVICES=0; \
python3 src/main.py \
--architecture dc --criterion mse --loss relativistic \
--data_path resources/celeba --logdir logs/dc_gan/m_r/
