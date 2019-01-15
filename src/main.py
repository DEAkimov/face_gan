from trainer import Trainer

if __name__ == '__main__':
    data_path = '../resources/celeba'
    logdir = '../logs/exp_0'
    write_period = 5
    batch_size, noise_size = 128, 128
    cuda = False
    n_epoch = 5

    trainer = Trainer(data_path, logdir, write_period,
                      batch_size, noise_size,
                      cuda)
    trainer.train(n_epoch)
