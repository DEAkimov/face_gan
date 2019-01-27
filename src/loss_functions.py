import torch
from torch.nn.functional import mse_loss


def update_statistics(criterion, dis_on_real, dis_on_fake):
    if criterion is mse_loss:
        dis_on_real = dis_on_real.mean().item()
        dis_on_fake = dis_on_fake.mean().item()
    else:
        dis_on_real = torch.sigmoid(dis_on_real).mean().item()
        dis_on_fake = torch.sigmoid(dis_on_fake).mean().item()
    return dis_on_real, dis_on_fake


# simple gan loss
def loss_dis(criterion, generator, discriminator,
             noise_size, real_data, device):
    batch_size = real_data.size(0)
    # discriminator on real
    dis_on_real = discriminator(real_data)  # D(x)
    real_labels = torch.full((batch_size,), 1, device=device)
    loss_d_real = criterion(dis_on_real, real_labels)

    # discriminator on fake
    noise = torch.randn(batch_size, noise_size, device=device)  # z
    with torch.no_grad():
        fake_data = generator(noise)  # G(z)
    dis_on_fake = discriminator(fake_data)  # D(G(z))
    fake_labels = torch.full((batch_size,), 0, device=device)
    loss_d_fake = criterion(dis_on_fake, fake_labels)
    loss_discriminator = loss_d_real + loss_d_fake

    # statistics
    dis_on_real, dis_on_fake = update_statistics(criterion, dis_on_real, dis_on_fake)
    return loss_discriminator, dis_on_real, dis_on_fake


def loss_gen(criterion, generator, discriminator,
             noise_size, real_data, device):
    batch_size = real_data.size(0)
    # generator
    noise = torch.randn(batch_size, noise_size, device=device)  # z
    gen_data = generator(noise)  # G(z)
    dis_on_gen = discriminator(gen_data)  # D(G(z))
    gen_labels = torch.full((batch_size,), 1, device=device)
    loss_generator = criterion(dis_on_gen, gen_labels)
    return loss_generator


# relativistic gan loss
def r_loss_dis(criterion, generator, discriminator,
               noise_size, real_data, device):
    batch_size = real_data.size(0)
    noise = torch.randn(batch_size, noise_size, device=device)  # z
    with torch.no_grad():
        fake_data = generator(noise)  # G(z)
    dis_on_real = discriminator(real_data)  # C(real)
    dis_on_fake = discriminator(fake_data)  # C(fake)

    real_labels = torch.full((batch_size,), 1, device=device)
    loss_discriminator = criterion((dis_on_real - dis_on_fake), real_labels)

    dis_on_real, dis_on_fake = update_statistics(criterion, dis_on_real, dis_on_fake)
    return loss_discriminator, dis_on_real, dis_on_fake


def r_loss_gen(criterion, generator, discriminator,
               noise_size, real_data, device):
    batch_size = real_data.size(0)
    noise = torch.randn(batch_size, noise_size, device=device)  # z
    gen_data = generator(noise)  # G(z)
    dis_on_real = discriminator(real_data)  # C(real)
    dis_on_gen = discriminator(gen_data)  # C(gen)
    gen_labels = torch.full((batch_size,), 1, device=device)

    loss_generator = criterion((dis_on_gen - dis_on_real), gen_labels)
    return loss_generator


# relativistic average gan loss
def ra_loss_dis(criterion, generator, discriminator,
                noise_size, real_data, device):
    batch_size = real_data.size(0)
    noise = torch.randn(batch_size, noise_size, device=device)  # z
    with torch.no_grad():
        fake_data = generator(noise)  # G(z)
    dis_on_real = discriminator(real_data)  # C(real)
    dis_on_fake = discriminator(fake_data)  # C(fake)

    real_labels = torch.full((batch_size,), 1, device=device)
    fake_labels = torch.full((batch_size,), 0, device=device)

    f_vs_r = dis_on_fake - dis_on_real.mean()  # fake more real than average real
    r_vs_f = dis_on_real - dis_on_fake.mean()  # real more real than average fake

    loss_discriminator = criterion(r_vs_f, real_labels) + criterion(f_vs_r, fake_labels)

    dis_on_real, dis_on_fake = update_statistics(criterion, dis_on_real, dis_on_fake)
    return loss_discriminator, dis_on_real, dis_on_fake


def ra_loss_gen(criterion, generator, discriminator,
                noise_size, real_data, device):
    batch_size = real_data.size(0)
    noise = torch.randn(batch_size, noise_size, device=device)  # z
    gen_data = generator(noise)  # G(z)

    dis_on_real = discriminator(real_data)  # C(real)
    dis_on_gen = discriminator(gen_data)  # C(gen)

    f_vs_r = dis_on_gen - dis_on_real.mean()  # gen more real than average real
    r_vs_f = dis_on_real - dis_on_gen.mean()  # real more real than average gen

    real_labels = torch.full((batch_size,), 1, device=device)
    fake_labels = torch.full((batch_size,), 0, device=device)

    # inverse for gen compared to dis
    loss_generator = criterion(f_vs_r, real_labels) + criterion(r_vs_f, fake_labels)
    return loss_generator
