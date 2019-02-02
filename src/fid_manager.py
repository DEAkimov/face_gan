import torch
import numpy as np
from scipy import linalg


class FIDManager:
    def __init__(self,
                 data_loader, noise_size,
                 generator, inception,
                 gpu_device):
        self.data_loader = data_loader
        self.noise_size = noise_size
        self.generator = generator
        self.inception = inception
        self.gpu_device = gpu_device

    @staticmethod
    def statistics_from_activations(activations):
        activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    @staticmethod
    def frechet_distance(mu1, sigma1, mu2, sigma2):
        # mu1, mu2 - mean over batch of the Inception last layer activation
        # sigma1, sigma2 - covariation matrices of the same variables
        # frechet_distance = ||mu_1 - mu_2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        diff = mu1 - mu2
        cov, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        cov = cov.real
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov)
        return fid

    def get_activations(self, mean, std):
        real_activations = []
        fake_activations = []
        for real, _ in self.data_loader:
            batch_size = real.size(0)
            real = real.to(self.gpu_device)
            with torch.no_grad():
                real_activations.append(self.inception(real))
                # should be ok even when generate whole batch once
                noise = torch.randn(
                    batch_size, 
                    self.noise_size,
                    device=self.gpu_device
                )
                fake = std.to(self.gpu_device) 
                       * self.generator(noise).to(self.gpu_device) 
                       + mean.to(self.gpu_device)
                fake_activations.append(self.inception(fake))
        return torch.cat(real_activations), torch.cat(fake_activations)

    def __call__(self, mean, std):
        self.generator.eval()
        real_activations, fake_activations = self.get_activations(mean, std)
        mu1, sigma1 = self.statistics_from_activations(real_activations)
        mu2, sigma2 = self.statistics_from_activations(fake_activations)
        frechet_distance = self.frechet_distance(mu1, sigma1, mu2, sigma2)
        return frechet_distance
