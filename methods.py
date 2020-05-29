import math
import torch
import torch.nn as nn
from model import FF


# refer to https://arxiv.org/abs/1801.04062
class MINE(nn.Module):
    def __init__(self, dim, hidden, layers, carry_rate=0.99):
        super(MINE, self).__init__()
        self.fXY = FF(2 * dim, hidden, 1, layers)
        self.carry_rate = carry_rate
        self.ema = None

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))
        infs = torch.tensor([float('inf')] * N).to(X.device)

        S = self.fXY(XY_package).view(N, N)
        joint = S.diag().mean()
        exp_marginal = (S - infs.diag()).exp().sum() / N / (N - 1)

        self.ema = exp_marginal.detach() if self.ema is None else \
                   self.carry_rate * self.ema + \
                   (1 - self.carry_rate) * exp_marginal.detach()

        mine_loss = (1 / self.ema) * exp_marginal - joint
        dv_loss = self.ema.log() - joint

        return (dv_loss - mine_loss).detach() + mine_loss

def setup_method(method, dim, hidden, layers):
    method = method.lower()
    assert method in ['mine','doe']

    if method in ['mine']:
        return MINE(dim, hidden, layers)