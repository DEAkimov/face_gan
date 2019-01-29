# GAN for face generation
Architectures: 
DCGAN, 
Self-Attention GAN (unconditioned), 
BigGAN (unconditioned)

Loss functions: 
simple GAN, Relativistic, Relativistic average

Criterions: Binary Cross Entropy (bce), Least Squares (LS), Hinge (H)

Conventionally, Hinge GAN optimize hinge loss only for discriminator. 
However, in this implementation, it optimizes hinge loss for generator too.

# work in progress

# Plan
- [x] GAN
- [x] LSGAN
- [x] Hinge loss
- [x] Relativistic GAN
- [x] Self-attention GAN
- [ ] Big GAN

BigGAN:
- [x] Hierarchical latent space
- [x] Architecture
- [x] orthogonal regularization
- [x] Truncated Normal for better sampling and better results
- [x] Full inception network for FID calculation
- [x] Training setup from BigGAN paper:
Adam 2e-4 in D and 5e-5 in G; β1 = 0 and β2 = 0.999; 2 D steps and 1 G step
- [x] Moving average of weights for better testing generation
- [x] DataParallel for training with adequate batch size
- [ ] Debug BigGAN =)

Test models:
- [ ] generate some samples for each model in the separate jupyter notebooks
- [ ] play with noise distribution
- [ ] find some patterns in the latent space