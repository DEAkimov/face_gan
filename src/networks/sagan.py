import torch
import torch.nn as nn


class SelfAttention(nn.Module):
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


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        sn = nn.utils.spectral_norm
        self.model = nn.Sequential(
            sn(nn.ConvTranspose2d(noise_size, 512, kernel_size=4, bias=False)),
            nn.BatchNorm2d(512), nn.ReLU(True),
            sn(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256), nn.ReLU(True),
            sn(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128), nn.ReLU(True),
            sn(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(64), nn.ReLU(True),

            sn(nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            SelfAttention(64),
            nn.BatchNorm2d(64), nn.ReLU(True),

            sn(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.Tanh()
        )
        self.init_print()

    def init_print(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Generator initialized. #parameters = {}'.format(num_params))

    def forward(self, input_noise):
        input_noise = input_noise.unsqueeze(-1).unsqueeze(-1)  # add H and W dimensions
        return self.model(input_noise)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        sn = nn.utils.spectral_norm
        self.conv = nn.Sequential(
            sn(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 512, 4, 4]
            sn(nn.Conv2d(512, 1, kernel_size=4, bias=False))
        )
        self.init_print()

    def init_print(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Discriminator initialized. #parameters = {}'.format(num_params))

    def forward(self, image):
        conv = self.conv(image)
        return conv.squeeze()  # remove H, W, C dimensions


if __name__ == '__main__':
    import torch
    ns = 128
    generator = Generator(ns)
    discriminator = Discriminator()
    batch = 7
    noise = torch.randn(batch, ns)
    generated = generator(noise)
    print(generated.size())
    discriminated = discriminator(generated)
    print(discriminated.size())