import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as sn


class BatchNorm(nn.Module):
    def __init__(self, in_channels, noise_size):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels, affine=False)
        self.noise_emb = nn.Linear(noise_size, in_channels * 2)  # no sn here

    def forward(self, input_features, noise):
        result = self.batch_norm(input_features)
        gamma, beta = self.noise_emb(noise).chunk(2, 1)  # 2 chunks along dim 1
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # add H and W dimensions
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * result + beta


class BlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, noise_size):
        super(BlockUp, self).__init__()
        self.bn1 = BatchNorm(in_channels, noise_size)
        self.conv1 = sn(nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1, bias=False))
        self.bn2 = BatchNorm(out_channels, noise_size)
        self.conv2 = sn(nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.residual = sn(nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1, bias=False))

    def forward(self, input_features, noise):
        output_features = self.bn1(input_features, noise)
        output_features = self.conv1(F.relu(output_features))
        output_features = self.bn2(output_features, noise)
        output_features = self.conv2(F.relu(output_features))
        residual = self.residual(input_features)
        return output_features + residual


class BlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockDown, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = sn(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = sn(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.residual = sn(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False))

    def forward(self, input_features):
        output_features = self.bn1(input_features)
        output_features = self.conv1(F.relu(output_features))
        output_features = self.bn2(output_features)
        output_features = self.conv2(F.relu(output_features))
        residual = self.residual(input_features)
        return output_features + residual


class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.conv1 = sn(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = sn(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, input_features):
        output_features = self.conv1(input_features)
        output_features = F.relu(self.bn1(output_features))
        output_features = self.conv2(output_features)
        output_features = self.bn2(output_features)
        return input_features + output_features
