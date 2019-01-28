# this is NOT actually BigGan,
# but BigGAN-like architecture with
# self-attention, same block structure,
# hierarchical latent space,
# but without any conditioning (which is sad)


import torch.nn as nn
from torch.nn.utils import spectral_norm as sn

from src.networks.big_layers import BlockUp, BlockDown, Block, SumPooling
from src.networks.self_attention import SelfAttention


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ch = ch = 96
        self.linear = sn(nn.Linear(20, 4 * 4 * 16 * ch))
        self.blocks = nn.ModuleList([
            BlockUp(16 * ch, 16 * ch, 20),
            BlockUp(16 * ch, 8 * ch, 20),
            BlockUp(8 * ch, 8 * ch, 20),
            BlockUp(8 * ch, 4 * ch, 20),
            BlockUp(4 * ch, 2 * ch, 20),
        ])
        self.attention = SelfAttention(2 * ch)
        self.last_block = BlockUp(2 * ch, 1 * ch, 8)

        self.final_layers = nn.Sequential(
            nn.BatchNorm2d(1 * ch), nn.ReLU(),
            nn.Conv2d(1 * ch, 3, 3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_noise):
        noise_chunks = input_noise.split(20, 1)
        out = self.linear(noise_chunks[0]).view(-1, 16 * self.ch, 4, 4)
        for block, noise in zip(self.blocks, noise_chunks[1:]):
            out = block(out, noise)
        out = self.attention(out)
        out = self.last_block(out, noise_chunks[-1])
        out = self.final_layers(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ch = 96
        self.conv = nn.Sequential(
            BlockDown(3, 1 * ch),
            BlockDown(1 * ch, 2 * ch),
            SelfAttention(2 * ch),
            BlockDown(2 * ch, 4 * ch),
            BlockDown(4 * ch, 8 * ch),
            BlockDown(8 * ch, 8 * ch),
            BlockDown(8 * ch, 16 * ch),
            Block(16 * ch),
        )
        self.final_layers = nn.Sequential(
            nn.ReLU(), SumPooling(),
            sn(nn.Linear(ch * 16, 1))
        )

    def forward(self, image):
        conv = self.conv(image)
        result = self.final_layers(conv)
        return result


if __name__ == '__main__':
    import torch

    def count_params(net):
        return sum(p.numel() for p in net.parameters())
    inp = torch.randn(1, 128)
    gen, dis = Generator(), Discriminator()
    print(count_params(gen), count_params(dis))  # (52, 51)M, (116, 115)M
    generated = gen(inp)
    discriminated = dis(generated)
    print(discriminated.size())
