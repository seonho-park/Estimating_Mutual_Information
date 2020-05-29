import math
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# This class is originally from https://github.com/karlstratos/doe
class Normal(object):
    def __init__(self, dim, rho, device):
        assert abs(rho) <= 1
        self.dim = dim
        self.rho = rho
        self.pdf = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))

    def I(self):
        num_nats = - self.dim / 2 * math.log(1 - math.pow(self.rho, 2)) \
                   if abs(self.rho) != 1.0 else float('inf')
        return num_nats

    def hY(self):
        return 0.5 * self.dim * math.log(2 * math.pi)

    def draw_samples(self, num_samples):
        X, ep = torch.split(self.pdf.sample((2 * num_samples,)), num_samples)
        Y = self.rho * X + math.sqrt(1 - math.pow(self.rho, 2)) * ep
        return X, Y

