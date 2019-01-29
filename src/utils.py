import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

from src.loss_functions import loss_dis, loss_gen, r_loss_dis, r_loss_gen, ra_loss_dis, ra_loss_gen
from src.networks.dcgan import Generator as DCGenerator, Discriminator as DCDiscriminator
from src.networks.sagan import Generator as SAGenerator, Discriminator as SADiscriminator
from src.networks.biggan import Generator as BigGenerator, Discriminator as BigDiscriminator


def hinge_loss(prediction, labels):
    # custom realization of hinge loss, support labels {0, 1}
    loss = 1.0 - (2.0 * labels - 1.0) * prediction
    loss = torch.max(torch.zeros_like(loss), loss).mean()
    return loss


criteria = {
    'bce': binary_cross_entropy_with_logits,
    'mse': mse_loss,
    'hinge': hinge_loss
}

loss_pairs = {
    'simple': (loss_dis, loss_gen),
    'relativistic': (r_loss_dis, r_loss_gen),
    'relativistic_a': (ra_loss_dis, ra_loss_gen)
}

networks = {
    'dc': (DCGenerator, DCDiscriminator),
    'sa': (SAGenerator, SADiscriminator),
    'big': (BigGenerator, BigDiscriminator)
}


def get_loader(path, train, batch_size, image_size, num_workers):
    # I believe there is no need of special transform
    # for Inception network: it will be done automatically
    # when .forward() called.
    # The only necessary transform [-1, 1] -> [0, 1]
    # moved to the Inception directly
    data_set = ImageFolder(path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: 2. * x - 1.)
                           ]))
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=train,
                             num_workers=num_workers)
    train_or_val = 'train' if train else 'val'
    print(
        '    {} data loader initialized, '
        'batch_size = {}, len(data_loader) = {}'.format(
            train_or_val, batch_size, len(data_loader)
        )
    )
    return data_loader


def get_networks(nets_type, noise_size, device):
    def count_params(net):
        return sum(p.numel() for p in net.parameters())

    gen, dis = networks[nets_type]
    gen, dis = gen(noise_size).to(device), dis().to(device)
    gen = DataParallel(gen)
    dis = DataParallel(dis)
    print(
        '    networks initialized, '
        '#params(gen) = {}, '
        '#params(dis) = {}'.format(
            count_params(gen), count_params(dis)
        )
    )
    return gen, dis


def moving_average(model_old, model_new, alpha=0.9999):
    for param_old, param_new in zip(
            model_old.parameters(),
            model_new.parameters()
    ):  # sad smile again!
        param_old.data = alpha * param_old.data + (1.0 - alpha) * param_new.data


def truncated_normal(
        n_samples, noise_size, device,
        threshold=1.0):
    samples = torch.randn(n_samples, noise_size, device=device)
    smaller = samples < - threshold
    bigger = samples > threshold
    while smaller.sum() != 0 or bigger.sum() != 0:
        new_samples = torch.randn(n_samples, noise_size, device=device)
        samples[smaller] = new_samples[smaller]
        samples[bigger] = new_samples[bigger]
        smaller = samples < - threshold
        bigger = samples > threshold
    return samples


def orthogonal_regularization(model, device):
    penalty = torch.tensor(0.0, dtype=torch.float32, device=device)
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            shape = param.shape[0]
            flatten = param.view(shape, -1)
            beta_squared = torch.mm(flatten, flatten.t())  # W^T.W
            ones = torch.ones(shape, shape) - torch.eye(shape)  # 1 - I
            penalty += ((beta_squared * ones) ** 2).sum()  # (||W^T.W x (1 - I)||_F)^2
    return penalty
