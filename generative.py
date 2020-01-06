import torch
from torch.autograd import grad
from random import random

from utils.data import max_req, req_to_instances, sample_states
from utils.nets import F, G
from utils.vis import scatter, line


# hyperparameters
num_epochs = 5e4
vis_iter = 200
batch_size = 256
latent_size = 4
f_epochs = 5
clip = 0.01
λ = 10
lr = 3e-5


# functions and such
def D(s):
    return torch.FloatTensor([req_to_instances(t.item()) for t in s]).unsqueeze(-1)
f = F(lr)
g = G(latent_size, lr)


# interpolation helper function
def interpolate(x, y):
    i = torch.rand_like(x)
    return (i * x) + ((1 - i) * y)


# norm helper function
def norm(x):
    return torch.sqrt(torch.sum(x.pow(2), dim=-1) + 1e-10)


# training
for epoch in range(int(num_epochs)):

    # sample new states for training during this epoch
    s = sample_states(batch_size)
    s_normalized = s / max_req

    # maximize f
    for _ in range(f_epochs):
        x = D(s)
        x_gen = g(s_normalized)
        
        # for p in f.parameters():
        #     p.data.clamp_(-clip, clip)
        # grad_penalty = 0

        x_inter = interpolate(x, x_gen)
        out = f(x_inter, s)
        grads = grad(out, x_inter, torch.ones(out.shape), create_graph=True)[0]
        grad_penalty = λ * ((norm(grads) - 1) ** 2).mean()

        f_loss = f(x, s_normalized).mean() - f(x_gen, s_normalized).mean() - grad_penalty
        f.maximize(f_loss)

    # minimize g
    g_loss = -f(g(s_normalized), s_normalized).mean()
    g.minimize(g_loss)

    # occasionally plot progress and example output
    if epoch % vis_iter == vis_iter - 1:
        line(epoch, f_loss.item(), 'f loss')
        line(epoch, g_loss.item(), 'g loss')

        points = []
        with torch.no_grad():
            s = []
            for i in range(0, max_req, max_req//20):
                s.extend(10 * [i])
            s = torch.FloatTensor(s).unsqueeze(-1)
            req = s.squeeze().tolist()
            ins = g(s / max_req).squeeze().tolist()

            points = list(zip(req,ins))
        scatter(points, 'Generative', clear=True)