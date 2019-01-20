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


def update_statistics(criterion, dis_on_real, dis_on_fake):
    if criterion is mse_loss:
        dis_on_real = dis_on_real.mean().item()
        dis_on_fake = dis_on_fake.mean().item()
    else:
        dis_on_real = torch.sigmoid(dis_on_real).mean().item()
        dis_on_fake = torch.sigmoid(dis_on_fake).mean().item()
    return dis_on_real, dis_on_fake


def get_loader(path, batch_size):
    data_set = ImageFolder(path,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: 2. * x - 1.)
                           ]))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader
