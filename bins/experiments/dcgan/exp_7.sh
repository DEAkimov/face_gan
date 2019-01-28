#!/bin/bash
export CUDA_VISIBLE_DEVICES=3; \
python3 src/main.py \
--architecture dc --criterion mse --loss relativistic_a \
--data_path resources/celeba --logdir logs/dc_gan/m_ra/
