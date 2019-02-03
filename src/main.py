import multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(project_dir)

from src.trainer import Trainer
from src.writer import Writer
from src.networks.inception import Inception
from src.fid_manager import FIDManager
from src.utils import criteria, loss_pairs, get_loader, get_networks, set_random_seed
from src.tqdm import disable_tqdm


if __name__ == '__main__':
    def bool_type(arg):
        if arg in ['True', 'true']:
            return True
        else:
            return False
    # args
    parser = argparse.ArgumentParser(description='GAN training runner')
    parser.add_argument("architecture", type=str,
                        help="networks architecture, one from {\'dc\', \'sa\', \'big\'}")
    parser.add_argument("criterion", type=str,
                        help="criterion type, one from {\'bce\', \'mse\', \'hinge\'}")
    parser.add_argument("loss", type=str,
                        help="loss type, one from {\'simple\', \'relativistic\', 'relativistic_a'}")
    parser.add_argument("data_path", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("--n_discriminator", type=int, default=2,
                        help='number of discriminator updates per one generator update, default=2')
    parser.add_argument("--write_period", type=int, default=5,
                        help='tensorboard write frequency, default=5')
    parser.add_argument("--fid_period", type=int, default=50,
                        help='fid calculation period, default=50')
    parser.add_argument("--batch_size", type=int, default=128,
                        help='DOUBLE batch size, default=128')
    parser.add_argument("--img_size", type=int, default=64,
                        help='image size, MUST be 64 or 128, default=64')
    parser.add_argument("--noise_size", type=int, default=128,
                        help='noise size, default=128')
    parser.add_argument("--orthogonal_penalty", type=float, default=1e-4,
                        help='orthogonal penalty coefficient, default=1e-4')
    parser.add_argument("--normalize", type=bool_type, default='True',
                        help='normalize training data if True, default=True')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='num_workers for data_loader, default=0')
    parser.add_argument("--n_epoch", type=int, default=5,
                        help='number of training epoch, default=5')
    parser.add_argument('--local_rank', type=int)
    args, remaining = parser.parse_known_args()

    local_rank = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    set_random_seed(42)

    if local_rank == 0:
        # some prints
        print('----------- Training script -----------')
        print('experiment settings:')
        print('    loss: {}'.format(args.loss))
        print('    criterion: {}'.format(args.criterion))
        print('    orthogonal: {}'.format(args.orthogonal_penalty))
        print('    normalize: {}'.format(args.normalize))
        print('    data_path: {}'.format(args.data_path))
        print('    logdir: {}'.format(args.logdir))
        print('initializations...')
    else:
        disable_tqdm()

    # initialize all necessary objects
    cuda = torch.cuda.is_available()
    # training on cpu will take approximately +inf time
    device = torch.device('cuda' if cuda else 'cpu')
    generator, discriminator, ma_generator = get_networks(
        args.architecture, args.noise_size, device
    )
    train_data = get_loader(
        args.data_path + '/train', True,
        args.batch_size, args.img_size,
        args.num_workers)
    val_data = get_loader(
        args.data_path + '/val', False,
        args.batch_size, args.img_size,
        args.num_workers)
    if local_rank == 0:
        writer = Writer(args.logdir, args.write_period)
    else:
        writer = Writer(args.logdir, args.write_period)
    inception = DistributedDataParallel(Inception().to(device))
    # measure performance of moving average generator
    fid_manager = FIDManager(
        val_data, args.noise_size,
        generator, ma_generator,
        inception, device
    )

    # initialize trainer
    trainer = Trainer(
        local_rank,
        generator, discriminator, ma_generator,
        train_data, val_data, fid_manager,
        criteria[args.criterion], loss_pairs[args.loss],
        args.n_discriminator, writer, args.logdir,
        args.write_period, args.fid_period,
        args.noise_size, args.orthogonal_penalty,
        args.normalize, device
    )
    if local_rank == 0:
        print('done')
    # training
    trainer.train(args.n_epoch)
