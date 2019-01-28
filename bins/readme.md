# Experiments description

All experiments conducted on celeba faces, downloaded from 
[here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

Baseline model (DCGAN, exp_0 - exp_5) trained to generate 64×64 images, 

- [x] exp_0: base experiment with all default parameters. 
criterion = bce, loss = simple, logdir = b_s
- [x] exp_1: criterion = mse, loss = simple, logdir = m_s
- [ ] exp_2: criterion = hinge, loss = simple, logdir = h_s

- [x] exp_3: criterion = bce, loss = relativistic, logdir = b_r
- [x] exp_4: criterion = mse, loss = relativistic, logdir = m_r
- [ ] exp_5: criterion = hinge, loss = relativistic, logdir = h_r
- [x] exp_6: criterion = bce, loss = relativistic_a, logdir = b_ra
- [x] exp_7: criterion = mse, loss = relativistic_a, logdir = m_ra
- [ ] exp_8: criterion = hinge, loss = relativistic_a, logdir = h_ra

Advanced models (SAGAN, exp_6 - exp_11; BigGAN, exp_12-17) 
trained to generate 128×128 images
