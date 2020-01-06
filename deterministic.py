import torch
from utils.data import plot_data, max_req, sample_states, req_to_instances
from utils.nets import Deterministic
from utils.vis import scatter, line

num_epochs = 1e4
vis_iter = 1000
batch_size = 512
lr = 3e-4

def D(s):
    return torch.FloatTensor([req_to_instances(t.item()) for t in s]).unsqueeze(-1)


f = Deterministic(lr)


# plot data at the beginning
plot_data()

# training
for epoch in range(int(num_epochs)):

    # sample new states and subsequent instance counts for training during this epoch
    s = sample_states(batch_size)
    y_hat = D(s)

    # expectation maximization
    loss = ((f(s / max_req) - y_hat) ** 2).mean()                     #! states are normalized to prevent NaN's during log_prob calculation
    f.minimize(loss)

    # occasionally plot progress and example output
    if epoch % vis_iter == vis_iter - 1:
        line(epoch, loss.item(), 'MSE loss')

        points = []
        with torch.no_grad():
            s = []
            for i in range(0, max_req, max_req//20):
                s.extend(10 * [i])
            s = torch.FloatTensor(s).unsqueeze(-1)
            req = s.squeeze().tolist()
            ins = f(s / max_req).squeeze().tolist()

            points = list(zip(req,ins))
        scatter(points, 'MSE', clear=True)