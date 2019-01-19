import argparse
from torch import cuda

from trainer import Trainer
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

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='GAN training runner')
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
    parser.add_argument("--noise_size", type=int, default=128,
                        help='noise size, default=128')
    parser.add_argument("--n_epoch", type=int, default=5,
                        help='number of training epoch, default=5')
    args, remaining = parser.parse_known_args()

    # training
    print('----------- Training script -----------')
    print('experiment settings:')
    print('    loss: {}'.format(args.loss))
    print('    criterion: {}'.format(args.criterion))
    print('    data_path: {}'.format(args.data_path))
    print('    logdir: {}'.format(args.logdir))
    trainer = Trainer(criteria[args.criterion], loss_pairs[args.loss],
                      args.data_path, args.logdir, args.write_period,
                      args.batch_size, args.noise_size,
                      cuda.is_available())
    trainer.train(args.n_epoch)
