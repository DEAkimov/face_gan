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

SAGAN:
- [x] implement self-attention layer
- [ ] implement architecture
- [ ] debug

BigGAN:
- [ ] Hierarchical latent space
- [ ] double channels number
- [ ] orthogonal regularization
