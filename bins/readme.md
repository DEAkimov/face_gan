# Experiments description

All experiments conducted on celeba faces, downloaded from 
[here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

Baseline model (DCGAN, exp_0 - exp_5) trained to generate 64×64 images, 

- [ ] exp_0: base experiment with all default parameters. 
criterion = bce, loss = simple, logdir = exp_0
- [ ] exp_1: change criterion to mse
- [ ] exp_2: criterion = bce, loss = relativistic
- [ ] exp_3: criterion = mse, loss = relativistic
- [ ] exp_4: criterion = bce, loss = relativistic_a
- [ ] exp_5: criterion = mse, loss = relativistic_a

Advanced models (SAGAN, exp_6 - exp_11; BigGAN, exp_12-17) 
trained to generate 128×128 images
