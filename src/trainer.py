from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from networks import Generator, Discriminator
from writer import Writer


def get_loader(path, batch_size):
    data_set = ImageFolder(path,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: 2. * x - 1.)
                           ]))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader


class Trainer:
    def __init__(self,
                 criterion, loss,
                 data_path, logdir, write_period,
                 batch_size, noise_size,
                 cuda):
        # nets, optimizers and criterion
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.generator = Generator(noise_size)
        self.discriminator = Discriminator()
        self.g_optim = Adam(self.generator.parameters(), 0.0002, (0.5, 0.999))
        self.d_optim = Adam(self.discriminator.parameters(), 0.0002, (0.5, 0.999))
        self.criterion = criterion
        self.loss_dis, self.loss_gen = loss

        # constants
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.logdir = logdir
        self.write_period = write_period

        # data and writer
        self.data_loader = get_loader(data_path, batch_size)
        self.writer = Writer(logdir, write_period)

        self.init_print()

    def init_print(self):
        print('Trainer initialized')

        print('    batch_size = {}'.format(self.batch_size))
        print('    noise_size = {}'.format(self.noise_size))
        print('    write_period = {}'.format(self.write_period))

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
        self.writer.update_statistics(
            loss_generator.item(),
            loss_discriminator.item(),
            dis_on_real, dis_on_fake
        )

        # write logs
        if step % self.write_period == 0:
            fake_data = self.generate_images(7*7)
            self.writer.write_logs(
                0.5 * (real_data[:7*7] + 1.0),
                0.5 * (fake_data + 1.0)
            )

    def train(self, n_epoch):
        print('start training for {} epoch')
        for epoch in range(n_epoch):
            for i, (real_data, _) in tqdm(
                    enumerate(self.data_loader, 0),
                    total=len(self.data_loader),
                    desc='epoch_{}'.format(epoch),
                    ncols=80
            ):  # sad smile
                self.train_step(i, real_data)
            self.save(self.logdir + '/checkpoint.pth')
        print('training done')
