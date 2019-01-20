import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from loss_functions import loss_dis, loss_gen, r_loss_dis, r_loss_gen, ra_loss_dis, ra_loss_gen
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

criteria = {
    'bce': binary_cross_entropy_with_logits,
    'mse': mse_loss
}

loss_pairs = {
    'simple': (loss_dis, loss_gen),
    'relativistic': (r_loss_dis, r_loss_gen),
    'relativistic_a': (ra_loss_dis, ra_loss_gen)
}


def get_loader(path, batch_size, image_size=64):
    data_set = ImageFolder(path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: 2. * x - 1.)
                           ]))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader
