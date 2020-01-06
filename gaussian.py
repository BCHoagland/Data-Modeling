import torch
from utils.data import plot_data, max_req, sample_states, req_to_instances
from utils.nets import Gaussian
from utils.vis import scatter, line

num_epochs = 1e4
vis_iter = 1000
batch_size = 512
lr = 3e-4

def D(s):
    return torch.FloatTensor([req_to_instances(t.item()) for t in s]).unsqueeze(-1)


π = Gaussian(lr)


# plot data at the beginning
plot_data()

# training
for epoch in range(int(num_epochs)):

    # sample new states and subsequent instance counts for training during this epoch
    s = sample_states(batch_size)
    y_hat = D(s)

    # expectation maximization
    objective = π.log_prob(s / max_req, y_hat).mean()                               # states are normalized to prevent NaN's during log_prob calculation
    π.maximize(objective)

    # occasionally plot progress and example output
    if epoch % vis_iter == vis_iter - 1:
        line(epoch, objective.item(), 'π objective')

        points = []
        with torch.no_grad():
            s = []
            for i in range(0, max_req, max_req//20):
                s.extend(10 * [i])
            s = torch.FloatTensor(s).unsqueeze(-1)
            req = s.squeeze().tolist()
            ins = π(s / max_req).squeeze().tolist()

            points = list(zip(req,ins))
        scatter(points, 'Gaussian', clear=True)