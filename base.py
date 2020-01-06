import torch
from random import random

from utils.data import plot_data, sample_data
from utils.nets import Base
from utils.vis import scatter, line

# plot data at the beginning
plot_data()

# training
net = Base()
for epoch in range(100):
    req, ins = sample_data()
    loss = net.optimize(req, ins)
    line(epoch, loss, 'loss - Base')

# final map
points = []
with torch.no_grad():
    req = torch.FloatTensor([int(random() * 200) for _ in range(100)]).unsqueeze(-1)
    ins = torch.FloatTensor([net(s) for s in req]).tolist()
    req = req.squeeze().tolist()

    points = list(zip(req,ins))
scatter(points, 'Base')