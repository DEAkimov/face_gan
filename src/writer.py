from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


class Writer:
    def __init__(self, logdir, write_period):
        self.write_period = write_period
        self.writer = SummaryWriter(logdir)

        # statistics
        self.writes_done = 0
        self.fids_done = 0
        self.loss_gen = 0.0
        self.loss_dis = 0.0
        self.p_real = 0.0
        self.p_fake = 0.0

    def _write_scalar(self, tag, scalar):
        self.writer.add_scalar(tag,
                               scalar / self.write_period,
                               self.writes_done)

    def _plot_images(self, real_data, fake_data):
        nrow = 3  # 7 for DC and SA GANs, 3 for Big
        self.writer.add_image('real', make_grid(real_data, nrow=nrow), self.writes_done)
        self.writer.add_image('fake', make_grid(fake_data, nrow=nrow), self.writes_done)

    def update_statistics(self, loss_gen, loss_dis, p_real, p_fake):
        self.loss_gen += loss_gen
        self.loss_dis += loss_dis
        self.p_real += p_real
        self.p_fake += p_fake

    def write_logs(self, real_data, fake_data):
        # write statistics
        self._write_scalar('loss/generator', self.loss_gen)
        self._write_scalar('loss/discriminator', self.loss_dis)
        self._write_scalar('p/real', self.p_real)
        self._write_scalar('p/fake', self.p_fake)
        # zero statistics
        self.loss_gen = 0.0
        self.loss_dis = 0.0
        self.p_real = 0.0
        self.p_fake = 0.0
        # plot images and update iterations
        self._plot_images(real_data, fake_data)
        self.writes_done += 1

    def write_fid(self, fid):
        self.writer.add_scalar('fid', fid, self.fids_done)
        self.fids_done += 1
