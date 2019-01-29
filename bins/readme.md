# Experiments description

All experiments conducted on celeba faces, downloaded from 
[here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

### DCGAN
exp_0 - exp_8, trained to generate 64×64 images

- [x] exp_0: base experiment with all default parameters. 
criterion = bce, loss = simple, logdir = b_s
- [x] exp_1: criterion = mse, loss = simple, logdir = m_s
- [x] exp_2: criterion = hinge, loss = simple, logdir = h_s
- [x] exp_3: criterion = bce, loss = relativistic, logdir = b_r
- [x] exp_4: criterion = mse, loss = relativistic, logdir = m_r
- [x] exp_5: criterion = hinge, loss = relativistic, logdir = h_r
- [x] exp_6: criterion = bce, loss = relativistic_a, logdir = b_ra
- [x] exp_7: criterion = mse, loss = relativistic_a, logdir = m_ra
- [x] exp_8: criterion = hinge, loss = relativistic_a, logdir = h_ra

checkpoint, logs and screenshots of the final results will be posted soon

### SAGAN
exp_9 - exp_17, trained to generate 128x128 images

- [x] exp_9: base exp; 10 epochs, criterion = bce, loss = simple, logdir = b_s
- [x] exp_10: mse, simple
- [x] exp_11: hinge, simple
- [x] exp_12: bce, relativistic
- [x] exp_13: mse, relativistic
- [x] exp_14: hinge, relativistic
- [x] exp_15: bse, relativistic_a
- [x] exp_16: mse, relativistic_a
- [x] exp_17: hinge, relativistic_a

### BigGAN
exp_18 - exp_26, trained to generate 256×256 images.
My model is unconditional, so I decide to not double networks size (channel multiplier = 64).
Only few loss functions will be tested: mse + simple, bce + ra, hinge + ra, mse + ra
