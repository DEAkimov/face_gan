#!/bin/bash

python3 src/main.py \
--architecture dc --criterion bce --loss simple \
--data_path resources/celeba --logdir logs/exp_0/
