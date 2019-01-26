from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

from src.loss_functions import loss_dis, loss_gen, r_loss_dis, r_loss_gen, ra_loss_dis, ra_loss_gen
from src.networks.dcgan import Generator as DCGenerator, Discriminator as DCDiscriminator
from src.networks.sagan import Generator as SAGenerator, Discriminator as SADiscriminator
from src.networks.biggan import Generator as BigGenerator, Discriminator as BigDiscriminator

criteria = {
    'bce': binary_cross_entropy_with_logits,
    'mse': mse_loss
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
    gen, dis = gen(noise_size), dis()
    gen = DataParallel(gen.to(device))
    dis = DataParallel(dis.to(device))
    print(
        '    networks initialized, '
        '#params(gen) = {}, '
        '#params(dis) = {}'.format(
            count_params(gen), count_params(dis)
        )
    )
    return gen, dis
