import os
import sys
import argparse
import torch
from copy import deepcopy

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(project_dir)

from src.trainer import Trainer
from src.writer import Writer
from src.networks.inception import Inception
from src.fid_manager import FIDManager
from src.utils import criteria, loss_pairs, get_loader, get_networks


if __name__ == '__main__':
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
    parser.add_argument("--num_workers", type=int, default=0,
                        help='num_workers for data_loader, default=0')
    parser.add_argument("--n_epoch", type=int, default=5,
                        help='number of training epoch, default=5')
    args, remaining = parser.parse_known_args()

    # some prints
    print('----------- Training script -----------')
    print('experiment settings:')
    print('    loss: {}'.format(args.loss))
    print('    criterion: {}'.format(args.criterion))
    print('    data_path: {}'.format(args.data_path))
    print('    logdir: {}'.format(args.logdir))
    print('initializations...')

    # initialize all necessary objects
    cuda = torch.cuda.is_available()
    gpu_device = torch.device('cuda' if cuda else 'cpu')
    cpu_device = torch.device('cpu')
    generator, discriminator = get_networks(args.architecture, args.noise_size, gpu_device)
    ma_generator = deepcopy(generator)  # moving average
    train_data = get_loader(
        args.data_path + '/train', True,
        args.batch_size, args.img_size,
        args.num_workers)
    val_data = get_loader(
        args.data_path + '/val', False,
        args.batch_size, args.img_size,
        args.num_workers)
    writer = Writer(args.logdir, args.write_period)
    inception = Inception().to(cpu_device)  # no need of DataParallel here
    # fid_manager = FIDManager(val_data, args.noise_size, generator, inception, device)
    # measure performance of moving average generator
    fid_manager = FIDManager(val_data, args.noise_size, ma_generator, inception, device)

    # initialize trainer
    trainer = Trainer(
        generator, discriminator, ma_generator,
        train_data, val_data, fid_manager,
        criteria[args.criterion], loss_pairs[args.loss],
        writer, args.logdir,
        args.write_period, args.fid_period,
        args.noise_size, gpu_device
    )
    print('done')
    # training
    trainer.train(args.n_epoch)
