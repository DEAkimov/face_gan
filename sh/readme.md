# Experiments desription

All experiments conducted on celeba faces, downloaded from 
[here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

Baseline model (DCGAN) trained to generate 64×64 images, 
and advanced model (SAGAN) trained to generate 128×128 images

- exp_0: base experiment with all default parameters. 
criterion = bce, loss = simple, logdir = exp_0
- exp_1: change criterion to mse
- exp_2: criterion = bce, loss = relativistic
- exp_3: criterion = mse, loss = relativistic
- exp_4: criterion = bce, loss = relativistic_a
- exp_5: criterion = mse, loss = relativistic_a