#!/bin/bash

python3 src/main.py \
--architecture dc --criterion bce --loss relativistic_a \
--data_path resources/celeba --logdir logs/dc_gan/exp_4/
