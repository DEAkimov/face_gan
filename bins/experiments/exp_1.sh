#!/bin/bash

python3 src/main.py \
--criterion mse --loss simple \
--data_path resources/celeba \
--logdir logs/test/
