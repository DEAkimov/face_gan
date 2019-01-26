# this is NOT actually BigGan,
# but BigGAN-like architecture with
# self-attention, same block structure,
# hierarchical latent space,
# but without any conditioning (which is sad)


import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, in_channels, noise_size):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels, affine=False)
        self.noise_emb = nn.Linear(noise_size, in_channels * 2)

    def forward(self, input_features, noise):
        result = self.batch_norm(input_features)
        gamma, beta = self.noise_emb(noise).chunk(2, 1)  # 2 chunks along dim 1
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # add H and W dimensions
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * result + beta


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, noise_size):
        super(Block, self).__init__()
        self.bn1 = BatchNorm(in_channels, noise_size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels, noise_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, input_features, noise):
        output_features = self.bn1(input_features, noise)
        output_features = self.conv1(F.relu(output_features))
        # TODO: up_sample / down_sample here?
        output_features = self.bn2(output_features, noise)
        output_features = self.conv2(F.relu(output_features))
        return output_features + input_features


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, input_noise):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, image):
        pass
