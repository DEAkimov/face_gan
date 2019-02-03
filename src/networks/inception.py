# needed to measure FID

import torch.nn as nn
from torchvision import models


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        # this super-simple copy-paste architecture works just fine
        # with 128x128 and 256x256 images, but fails with 64x64,
        # so I cut off the last block for 64x64
        # (and 128x128, which is a mistake but ok)
        inception = models.inception_v3(pretrained=True, transform_input=True)
        self.net = nn.Sequential(
            # block 0
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
            # block 1
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            # block 2
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            # block 3 - commented for 64x64 and 128x128 images
            # inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            # Keep avg_pool
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.net.eval()
        self.init_print()

    def init_print(self):
        num_params = sum(p.numel() for p in self.net.parameters())
        print('    inception initialized, #params = {}'.format(num_params))

    def forward(self, image):
        # as mentioned in the src.utils.get_loader,
        # transform [-1, +1] called just before net here
        image = 0.5 * (image + 1.0)
        net_out = self.net(0.5 * (image + 1.0))
        size = 768  # 768 for DC and SA GANs, 2048 for BigGAN
        return net_out.view(-1, size)


if __name__ == '__main__':
    import torch
    net = Inception()
    fake_images = [
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 128, 128),
        torch.randn(1, 3, 64, 64),  # not ok with full* inception, but ok with 'truncated' version
        # * full means without the last fully connected classification layer
    ]
    for fi in fake_images:
        net_output = net(fi)
        print(fi.size(), net_output.size())
