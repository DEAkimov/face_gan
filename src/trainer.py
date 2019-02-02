from tqdm import tqdm
import torch
from torch.optim import Adam

from src.utils import truncated_normal, moving_average, orthogonal_regularization


class Trainer:
    def __init__(self,
                 generator, discriminator, ma_generator,
                 train_data, val_data, fid_manager,
                 criterion, loss, n_discriminator,
                 log_writer, logdir,
                 write_period, fid_period,
                 noise_size, orthogonal_penalty,
                 gpu_device):
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
        lr_g, lr_d = 5e-5, 2e-4
        beta1, beta2 = 0.0, 0.999
        self.g_optim = Adam(self.generator.parameters(), lr_g, (beta1, beta2))
        self.d_optim = Adam(self.discriminator.parameters(), lr_d, (beta1, beta2))
        self.criterion = criterion
        self.loss_dis, self.loss_gen = loss

        # data and writer
        self.train_data_len = len(train_data)
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

    def generate_images(self, n_images):
        # self.generator.eval()
        self.ma_generator.eval()
        # truncated normal noise special for BigGAN
        noise = truncated_normal(n_images, self.noise_size, device=self.gpu_device)  # z
        # noise = torch.randn(n_images, self.noise_size, device=self.device)
        with torch.no_grad():
            # generated_images = self.generator(noise)  # G(z)
            # generate samples from moving_average model
            generated_images = self.ma_generator(noise)  # G(z)
        return generated_images

    def call_loss(self, loss, real_data):
        return loss(
            self.criterion, self.generator, self.discriminator,
            self.noise_size, real_data, self.gpu_device
        )

    def train_discriminator(self, real_data):
        self.discriminator.train()
        self.d_optim.zero_grad()
        loss_discriminator, dis_on_real, dis_on_fake = self.call_loss(self.loss_dis, real_data)
        loss_discriminator.backward()
        self.d_optim.step()
        return loss_discriminator.item(), dis_on_real, dis_on_fake

    def train_generator(self, real_data):
        self.generator.train()
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
            real_data = real_data.to(self.gpu_device)

            loss_d, dis_on_real, dis_on_fake = self.train_discriminator(real_data)

            s_loss_discriminator += loss_d / self.n_discriminator
            s_dis_on_real += dis_on_real / self.n_discriminator
            s_dis_on_fake += dis_on_fake / self.n_discriminator

        real_data, _ = next(data_loader)
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
        fake_data = self.generate_images(n_images * n_images)
        self.log_writer.write_logs(
            0.5 * (fake_data + 1.0),
        )

    def write_fid(self):
        fid = self.fid_manager()
        self.log_writer.write_fid(fid)

    def train(self, n_epoch):
        print('start training for {} epoch'.format(n_epoch))
        for epoch in range(n_epoch):
            data_loader = iter(self.train_data)
            for step in tqdm(
                    range(self.train_data_len // (self.n_discriminator + 1)),
                    desc='epoch_{}'.format(epoch),
                    ncols=90
            ):  # sad smile
                self.train_step(data_loader)
                if step % self.write_period == 0:
                    self.write_logs()
                if step % self.fid_period == 0:
                    self.write_fid()
            self.save(self.logdir + '/checkpoint.pth')
        print('training done')
