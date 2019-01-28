#!/bin/bash
export CUDA_VISIBLE_DEVICES=2; \
python3 src/main.py \
--architecture dc --criterion bce --loss relativistic_a \
--data_path resources/celeba --logdir logs/dc_gan/b_ra/
