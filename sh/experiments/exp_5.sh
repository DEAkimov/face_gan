#!/bin/bash

python3 src/main.py \
--criterion mse --loss relativistic_a \
--data_path resources/celeba \
--logdir logs/exp_5/
