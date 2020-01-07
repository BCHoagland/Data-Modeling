import torch
from math import sqrt, sin, cos
from random import random

from utils.vis import scatter

max_req = 400

def req_to_instances(req):
    # sqrt
    # mean = int(sqrt(req)) * 3
    # err = int(random() * 10 - 5)

    # fountain spray
    mean = -((req/20 - 10)**2) + 100
    err = int(random() * 10 - 5) * 10 * req / max_req
    
    return max(mean + err, 0)


def sample_states(batch_size=128):
    return torch.FloatTensor([int(random() * max_req) for _ in range(batch_size)]).unsqueeze(-1)


def sample_data(batch_size=128):
    s = sample_states(batch_size)
    a = torch.FloatTensor([req_to_instances(req) for req in s]).unsqueeze(-1)
    return s, a


def plot_data():
    points = []
    for req in range(0, max_req, max_req//20):
        for _ in range(10):
            ins = req_to_instances(req)
            points.append((req, ins))
    scatter(points, 'data')