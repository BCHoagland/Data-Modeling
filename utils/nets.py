import torch
import torch.nn as nn
from torch.distributions import Normal


class F(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1 + 1, 64),                                           #! generalize better
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, s):
        inp = torch.cat([x, s], dim=-1)
        return self.main(inp)
    
    def maximize(self, loss):
        self.optim.zero_grad()
        (-loss).backward()
        self.optim.step()


class G(nn.Module):
    def __init__(self, latent_size, lr):
        super().__init__()

        self.latent_size = latent_size

        self.main = nn.Sequential(
            nn.Linear(latent_size + 1, 64),                                 #! generalize better
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, s):
        batch_size = s.shape[0]
        z = torch.randn(batch_size, self.latent_size)
        inp = torch.cat([z, s], dim=-1)
        return self.main(inp)
    
    def minimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


class Gaussian(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.mean = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.log_std = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def dist(self, s):
        mean = self.mean(s)
        std = torch.exp(self.log_std(s))
        return Normal(mean, std)
    
    def forward(self, s):
        dist = self.dist(s)
        return dist.sample()
    
    def log_prob(self, s, a):
        dist = self.dist(s)
        return dist.log_prob(a) + 1e-10
    
    def maximize(self, loss):
        self.optim.zero_grad()
        (-loss).backward()
        self.optim.step()


class Deterministic(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, s):
        return self.main(s)
    
    def minimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()