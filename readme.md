# Plan
- [x] GAN
- [x] LSGAN
- [x] Relativistic GAN
- [x] Self-attention GAN

BigGAN:
- [ ] Hierarchical latent space
- [ ] double channels number
- [ ] orthogonal regularization

# TODO
- [ ] organize parameters and experiments for SAGAN
- [x] Inception Score is unacceptable because it designed for conditional gan, which is not our case 
- [ ] FID

# TODO for today:
- [x] MASSIVE refactor
- [x] split celeba into train and val data
- [x] load pre-trained inception
- [x] pre-pooling layer from inception
- [x] write FID algorithm and manager
- [ ] develop a schedule for measure FID

# TODO for tomorrow
- [ ] check if FID calculates correctly
- [ ] scheduler from yesterday
- [ ] finish BigGAN architecture
- [ ] run some experiments