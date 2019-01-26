import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    # TODO: check shapes
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        sn = nn.utils.spectral_norm

        self.k_projector = sn(nn.Conv2d(in_channels, in_channels // 8, 1))  # W_f
        self.q_projector = sn(nn.Conv2d(in_channels, in_channels // 8, 1))  # W_g
        self.v_projector = sn(nn.Conv2d(in_channels, in_channels, 1))  # W_h
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch, channels, height, width = x.size()
        key = self.k_projector(x).view(batch, channels // 8, height * width)
        query = self.q_projector(x).view(batch, channels // 8, height * width)
        value = self.v_projector(x).view(batch, channels, height * width)
        attn_logits = torch.matmul(key.permute(0, 2, 1), query)
        attn_weights = torch.softmax(attn_logits, dim=1)  # softmax over N dimension in key tensor
        context = torch.matmul(value, attn_weights).view(batch, channels, height, width)
        return x + self.gamma * context
