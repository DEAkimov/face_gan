import torch


class FDManager:
    def __init__(self,
                 data_loader, noise_size,
                 generator, inception, device):
        self.data_loader = data_loader
        self.noise_size = noise_size
        self.generator = generator
        self.inception = inception
        self.device = device

    @staticmethod
    def statistics_from_activations(activations):
        mean = activations.mean(0)
        activations_n = activations - mean
        cov = (activations_n.t() @ activations_n) / (activations.size(0) - 1)
        return mean, cov

    @staticmethod
    def frechet_distance(mu1, sigma1, mu2, sigma2):
        # mu1, mu2 - mean over batch of the Inception last layer activation
        # sigma1, sigma2 - covariation matrices of the same variables
        # frechet_distance = ||mu_1 - mu_2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        diff = mu1 - mu2
        mul = sigma1 @ sigma2
        mul[mul < 0] = 0.0  # sqrt return nan if there is values less than 0
        mul = torch.sqrt(mul)
        fd = torch.dot(diff, diff) + torch.trace(sigma1 + sigma2 - 2 * mul)
        return fd

    def get_activations(self):
        real_activations = []
        fake_activations = []
        for real in self.data_loader:
            batch_size = real.size(0)
            with torch.no_grad():
                noise = torch.randn(
                    batch_size,
                    self.noise_size,
                    device=self.device
                )
                fake = self.generator(noise)
                fake_activations.append(self.inception(fake))
                real_activations.append(self.inception(real))
        return torch.cat(real_activations), torch.cat(fake_activations)

    def __call__(self):
        real_activations, fake_activations = self.get_activations()
        mu1, sigma1 = self.statistics_from_activations(real_activations)
        mu2, sigma2 = self.statistics_from_activations(fake_activations)
        frechet_distance = self.frechet_distance(mu1, sigma1, mu2, sigma2)
        return frechet_distance
