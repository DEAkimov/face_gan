import torch
from torch.optim import Adam

from src.utils import truncated_normal, moving_average, orthogonal_regularization, data_statistics
from src.tqdm import tqdm


class Trainer:
    def __init__(self,
                 local_rank,
                 generator, discriminator, ma_generator,
                 train_data, val_data, fid_manager,
                 criterion, loss, n_discriminator,
                 log_writer, logdir,
                 write_period, fid_period,
                 noise_size, orthogonal_penalty,
                 normalize, gpu_device):
        self.local_rank = local_rank
        # nets, optimizers and criterion
        self.gpu_device = gpu_device
        self.generator = generator
        self.discriminator = discriminator
        self.ma_generator = ma_generator
        self.fid_manager = fid_manager
        # =============== parameters for DCGAN and SAGAN ===============
        # lr_g, lr_d = 4e-4, 1e-4
        # beta1, beta2 = 0.0, 0.9
        # =============== parameters for BigGAN ===============
        lr_g, lr_d = 2e-4, 5e-5
        beta1, beta2 = 0.0, 0.999
        self.g_optim = Adam(self.generator.parameters(), lr_g, (beta1, beta2))
        self.d_optim = Adam(self.discriminator.parameters(), lr_d, (beta1, beta2))
        self.criterion = criterion
        self.loss_dis, self.loss_gen = loss

        # data and writer
        self.train_data_len = len(train_data)
        if normalize:
            self.train_mean, self.train_std = data_statistics(val_data)
        else:
            self.train_mean = torch.tensor(0.0, dtype=torch.float32)
            self.train_std = torch.tensor(1.0, dtype=torch.float32)
        self.train_data = train_data
        self.val_data = val_data
        self.log_writer = log_writer

        # constants
        self.noise_size = noise_size
        self.n_discriminator = n_discriminator
        self.orthogonal_penalty = orthogonal_penalty
        self.logdir = logdir
        self.write_period = write_period
        self.fid_period = fid_period

        if self.local_rank == 0:
            self.init_print()

    def init_print(self):
        print('    trainer initialized')
        print('        noise_size = {}'.format(self.noise_size))
        print('        write_period = {}'.format(self.write_period))

    def save(self, filename):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

    def normalize(self, tensor):
        return (tensor - self.train_mean) / self.train_std

    def denormalize(self, tensor):
        tensor = self.train_std.to(self.gpu_device) * tensor
        tensor = tensor + self.train_mean.to(self.gpu_device)
        return tensor

    def generate_images(self, n_images):
        # truncated normal noise special for BigGAN
        generated_images = []
        generated_images_ma = []
        for i in range(n_images):
            n_noise = torch.randn(1, self.noise_size, device=self.gpu_device)
            t_noise = truncated_normal(1, self.noise_size, device=self.gpu_device)
            with torch.no_grad():
                generated = self.denormalize(self.generator(n_noise))
                generated_ma = self.denormalize(self.ma_generator(t_noise))
                generated_images.append(generated)
                generated_images_ma.append(generated_ma)
        return torch.cat(generated_images), torch.cat(generated_images_ma)

    def call_loss(self, loss, real_data):
        return loss(
            self.criterion, self.generator, self.discriminator,
            self.noise_size, real_data, self.gpu_device
        )

    def train_discriminator(self, real_data):
        self.d_optim.zero_grad()
        loss_discriminator, dis_on_real, dis_on_fake = self.call_loss(self.loss_dis, real_data)
        loss_discriminator.backward()
        self.d_optim.step()
        return loss_discriminator.item(), dis_on_real, dis_on_fake

    def train_generator(self, real_data):
        self.g_optim.zero_grad()
        loss_generator = self.call_loss(self.loss_gen, real_data)
        if self.orthogonal_penalty:
            penalty = orthogonal_regularization(self.generator, self.gpu_device)
            loss_generator = loss_generator + 1e-4 * penalty
        loss_generator.backward()
        self.g_optim.step()
        return loss_generator.item()

    def train_step(self, data_loader):
        # s stands for 'statistic'
        s_loss_discriminator, s_dis_on_real, s_dis_on_fake = 0.0, 0.0, 0.0
        for _ in range(self.n_discriminator):  # train discriminator for n steps
            real_data, _ = next(data_loader)
            real_data = self.normalize(real_data)
            real_data = real_data.to(self.gpu_device)

            loss_d, dis_on_real, dis_on_fake = self.train_discriminator(real_data)

            s_loss_discriminator += loss_d / self.n_discriminator
            s_dis_on_real += dis_on_real / self.n_discriminator
            s_dis_on_fake += dis_on_fake / self.n_discriminator

        real_data, _ = next(data_loader)
        real_data = self.normalize(real_data)
        real_data = real_data.to(self.gpu_device)

        loss_generator = self.train_generator(real_data)  # train generator once

        # update moving average generator
        moving_average(self.ma_generator, self.generator)

        return loss_generator, s_loss_discriminator, s_dis_on_real, s_dis_on_fake

    def update_statistics(self,
                          loss_generator,
                          loss_discriminator,
                          dis_on_real, dis_on_fake):
        self.log_writer.update_statistics(
            loss_generator,
            loss_discriminator,
            dis_on_real, dis_on_fake
        )

    def write_logs(self):
        n_images = 3  # 7 for DC and SA GANs, 3 for Big
        fake_data, fake_data_ma = self.generate_images(n_images * n_images)
        self.log_writer.write_logs(
            0.5 * (fake_data + 1.0),
            0.5 * (fake_data_ma + 1.0)
        )

    def write_fid(self):
        fid, fid_ma = self.fid_manager(self.train_mean, self.train_std)
        self.log_writer.write_fid(fid, fid_ma)

    def train(self, n_epoch):
        print('start training for {} epoch'.format(n_epoch))
        self.discriminator.train()
        self.generator.train()
        self.ma_generator.eval()
        step = 0
        for epoch in range(n_epoch):
            data_loader = iter(self.train_data)
            for i in tqdm(
                    range(self.train_data_len // (self.n_discriminator + 1)),
                    desc='epoch_{}'.format(epoch),
                    ncols=90
            ):  # sad smile
                step_statisctics = self.train_step(data_loader)
                # self.update_statistics(*step_statisctics)
                # if step % self.write_period == 0:
                #     self.write_logs()
                # if step % self.fid_period == 0:
                #     self.write_fid()
                step += 1
            self.save(self.logdir + '/checkpoint.pth')
        print('training done')
