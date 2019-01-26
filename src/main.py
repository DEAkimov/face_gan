import os
import sys
import argparse
import torch

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(project_dir)

from src.trainer import Trainer
from src.writer import Writer
from src.fid_manager import FIDManager
from src.utils import criteria, loss_pairs, get_loader, get_networks


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='GAN training runner')
    parser.add_argument("architecture", type=str,
                        help="networks architecture, one from {\'dc\', \'sa\', \'big\'}")
    parser.add_argument("criterion", type=str,
                        help="criterion type, one from {\'bce\', \'mse\'}")
    parser.add_argument("loss", type=str,
                        help="loss type, one from {\'simple\', \'relativistic\', 'relativistic_a'}")
    parser.add_argument("data_path", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("--write_period", type=int, default=5,
                        help='tensorboard write frequency, default=5')
    parser.add_argument("--batch_size", type=int, default=128,
                        help='batch size, default=128')
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
    device = torch.device('cuda' if cuda else 'cpu')
    generator, discriminator = get_networks(args.architecture, args.noise_size, device)
    train_data = get_loader(
        args.data_path, True,
        args.batch_size, args.img_size,
        args.num_workers)
    val_data = get_loader(
        args.data_path + '/val', False,
        args.batch_size, args.img_size,
        args.num_workers)
    writer = Writer(args.logdir, args.write_period)
    fid_manager = FIDManager()  # TODO

    # initialize trainer
    trainer = Trainer(
        generator, discriminator,
        train_data, val_data, writer,
        criteria[args.criterion], loss_pairs[args.loss],
        args.logdir, args.write_period,
        args.noise_size,
        device
    )
    print('done')
    # training
    trainer.train(args.n_epoch)
