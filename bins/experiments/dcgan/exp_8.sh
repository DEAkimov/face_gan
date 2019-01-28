#!/bin/bash
export CUDA_VISIBLE_DEVICES=0; \
python3 src/main.py \
--architecture dc --criterion hinge --loss relativistic_a \
--data_path resources/celeba --logdir logs/dc_gan/h_ra/
