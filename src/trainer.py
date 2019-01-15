from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

from networks import Generator, Discriminator


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
                 data_path, logdir, write_period,
                 batch_size, noise_size=128,
                 cuda=True):
        # nets, optimizers and criterion
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.generator = Generator(noise_size)
        self.discriminator = Discriminator()
        self.g_optim = Adam(self.generator.parameters(), 0.0002, (0.5, 0.999))
        self.d_optim = Adam(self.discriminator.parameters(), 0.0002, (0.5, 0.999))
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # constants
        self.real_label = 1
        self.fake_label = 0
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.logdir = logdir
        self.write_period = write_period

        # data and writer
        self.data_loader = get_loader(data_path, batch_size)
        self.writer = SummaryWriter(logdir)

        # statistics
        self.iterations_done = 0
        self.loss_gen = 0.0
        self.loss_dis = 0.0
        self.log_p_real = 0.0
        self.log_p_fake = 0.0

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

    def write_scalar(self, tag, scalar):
        self.writer.add_scalar(tag,
                               scalar / self.write_period,
                               self.iterations_done)

    def write_logs(self):
        # write statistics
        self.write_scalar('loss/generator', self.loss_gen)
        self.write_scalar('loss/discriminator', self.loss_dis)
        self.write_scalar('log_p/real', self.log_p_real)
        self.write_scalar('log_p/fake', self.log_p_fake)
        # zero statistics
        self.loss_gen = 0.0
        self.loss_dis = 0.0
        self.log_p_real = 0.0
        self.log_p_fake = 0.0

    def plot_images(self, real_data, fake_data):
        self.writer.add_image('real', make_grid(real_data), self.iterations_done)
        self.writer.add_image('fake', make_grid(fake_data), self.iterations_done)

    def optimize_generator(self, loss):
        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()

    def optimize_discriminator(self, loss):
        self.d_optim.zero_grad()
        loss.backward()
        self.d_optim.step()

    # simple gan
    def train_on_batch(self, real_data):
        # discriminator on real
        dis_on_real = self.discriminator(real_data)  # D(x)
        real_labels = torch.full((self.batch_size,),
                                 self.real_label,
                                 device=self.device)
        loss_d_real = self.criterion(dis_on_real, real_labels)

        # discriminator on fake
        noise = torch.randn(self.batch_size, self.noise_size)  # z
        fake_data = self.generator(noise)  # G(z)
        dis_on_fake = self.discriminator(fake_data.detach())  # D(G(z))
        fake_labels = torch.full((self.batch_size,),
                                 self.fake_label,
                                 device=self.device)
        loss_d_fake = self.criterion(dis_on_fake, fake_labels)
        loss_discriminator = loss_d_real + loss_d_fake
        self.optimize_discriminator(loss_discriminator)

        # generator
        dis_on_gen = self.discriminator(fake_data)
        loss_g = self.criterion(dis_on_gen, real_labels)
        self.optimize_generator(loss_g)

        # update statistics
        self.loss_dis += loss_discriminator.item()
        self.log_p_real += torch.sigmoid(dis_on_real).mean()
        self.log_p_fake += torch.sigmoid(dis_on_fake).mean()
        self.loss_gen += loss_g.item()
        return fake_data

    def train(self, n_epoch):
        print('start training')
        for epoch in range(n_epoch):
            for i, (real_data, _) in tqdm(
                    enumerate(self.data_loader, 0),
                    total=len(self.data_loader),
                    desc='epoch_{}'.format(epoch)
            ):  # sad smile
                fake_data = self.train_on_batch(real_data)
                if i % self.write_period == 0:
                    self.write_logs()
                    self.plot_images(0.5 * (real_data + 1.0), 0.5 * (fake_data + 1.0))
                    self.iterations_done += 1
            self.save(self.logdir + '/checkpoint.pth')
        print('training done')
