#!/bin/bash

python3 src/main.py \
--criterion bce --loss relativistic \
--data_path resources/celeba \
--logdir logs/exp_2/
