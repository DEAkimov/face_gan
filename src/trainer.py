from tqdm import tqdm
import torch
from torch.optim import Adam


class Trainer:
    def __init__(self,
                 generator, discriminator,
                 train_data, val_data, fid_manager,
                 criterion, loss,
                 log_writer, logdir,
                 write_period, fid_period,
                 noise_size, device):
        # nets, optimizers and criterion
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.fid_manager = fid_manager
        self.g_optim = Adam(self.generator.parameters(), 1e-4, (0.0, 0.9))
        self.d_optim = Adam(self.discriminator.parameters(), 4e-4, (0.0, 0.9))
        self.criterion = criterion
        self.loss_dis, self.loss_gen = loss

        # constants
        self.noise_size = noise_size
        self.logdir = logdir
        self.write_period = write_period
        self.fid_period = fid_period

        # data and writer
        self.train_data = train_data
        self.val_data = val_data
        self.log_writer = log_writer

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
        noise = torch.randn(n_images, self.noise_size)  # z
        with torch.no_grad():
            generated_images = self.generator(noise)  # G(z)
        return generated_images

    def optimize_gen(self, loss_generator):
        self.g_optim.zero_grad()
        loss_generator.backward()
        self.g_optim.step()

    def optimize_dis(self, loss_discriminator):
        self.d_optim.zero_grad()
        loss_discriminator.backward()
        self.d_optim.step()

    def call_loss(self, loss, real_data):
        return loss(
            self.criterion, self.generator, self.discriminator,
            self.noise_size, real_data, self.device
        )

    def train_step(self, step, real_data):
        # compute and optimize losses
        loss_discriminator, dis_on_real, dis_on_fake = self.call_loss(self.loss_dis, real_data)
        self.optimize_dis(loss_discriminator)
        loss_generator = self.call_loss(self.loss_gen, real_data)
        self.optimize_gen(loss_generator)

        # update statistics
        self.log_writer.update_statistics(
            loss_generator.item(),
            loss_discriminator.item(),
            dis_on_real, dis_on_fake
        )

        # write logs
        if step % self.write_period == 0:
            fake_data = self.generate_images(7*7)
            self.log_writer.write_logs(
                0.5 * (real_data[:7*7] + 1.0),
                0.5 * (fake_data + 1.0),
            )
        # write fid
        if step % self.fid_period == 0:
            fid = self.fid_manager()
            self.log_writer.write_fid(fid)

    def train(self, n_epoch):
        print('start training for {} epoch'.format(n_epoch))
        for epoch in range(n_epoch):
            for i, (real_data, _) in tqdm(
                    enumerate(self.train_data, 0),
                    total=len(self.train_data),
                    desc='epoch_{}'.format(epoch),
                    ncols=90
            ):  # sad smile
                self.train_step(i, real_data.to(self.device))
            self.save(self.logdir + '/checkpoint.pth')
        print('training done')
