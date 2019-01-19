import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_size, 512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

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
            sn(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),  # [batch, 512, 4, 4]
            sn(nn.Conv2d(512, 1, kernel_size=4, bias=False))
        )

    def forward(self, image):
        conv = self.conv(image)
        return conv.squeeze()  # remove H, W, C dimensions


if __name__ == '__main__':
    import torch
    noise_size = 128
    generator = Generator(noise_size)
    discriminator = Discriminator()
    batch = 7
    noise = torch.randn(batch, noise_size)
    generated = generator(noise)
    print(generated.size())
    discriminated = discriminator(generated)
    print(discriminated.size())
