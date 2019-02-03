import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
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
    local_rank = dist.get_rank()
    data_set = ImageFolder(path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: 2. * x - 1.)
                           ]))
    sampler = DistributedSampler(
        data_set, 
        num_replicas=dist.get_world_size(),
        rank=local_rank
    )
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers)
    train_or_val = 'train' if train else 'val'
    if local_rank == 0:
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

    local_rank = dist.get_rank()
    gen, dis = networks[nets_type]
    gen, dis = gen(noise_size).to(device), dis().to(device)
    gen_dp = DistributedDataParallel(gen, device_ids=[local_rank])
    dis_dp = DistributedDataParallel(dis, device_ids=[local_rank])
    gen_ma = DistributedDataParallel(deepcopy(gen), device_ids=[local_rank])
    if local_rank == 0:
        print(
            '    networks initialized, '
            '#params(gen) = {}, '
            '#params(dis) = {}'.format(
                count_params(gen), count_params(dis)
            )
        )
    return gen_dp, dis_dp, gen_ma


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
    # (||W^T.W x (1 - I)||_F)^2
    penalty = torch.tensor(0.0, dtype=torch.float32, device=device)
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            shape = param.shape[0]
            flatten = param.view(shape, -1)
            beta_squared = torch.mm(flatten, flatten.t())  # W^T.W
            ones = torch.ones(shape, shape, dtype=torch.float32)
            diag = torch.eye(shape, dtype=torch.float32)
            penalty += ((beta_squared * (ones - diag).to(device)) ** 2).sum()
    return penalty


def data_statistics(data_loader):
    print('calculating data statistics...')
    sample = data_loader.dataset[0][0]
    mean = torch.zeros_like(sample)
    var = torch.zeros_like(sample)
    for (x, _) in tqdm(data_loader):
        mean += x.mean(0)
        var += (x ** 2).mean(0)
    mean = mean / len(data_loader)
    total_size = len(data_loader) * data_loader.batch_size
    var = var / len(data_loader) - mean ** 2
    var = (total_size / (total_size - 1)) * var
    std = torch.sqrt(var)
    return mean, std


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sync(device):
    sync_tensor = torch.tensor(0, device=device)
    dist.all_reduce(sync_tensor)
