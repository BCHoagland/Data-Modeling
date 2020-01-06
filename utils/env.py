from utils.data import req_to_instances, sample_states

# negative rewards gets larger in magnitude as the given action gets further away from a sampled optimal action
# there's some randomness here! this function is stochastic if req_to_instances is stochastic
def reward(s, a):
    a_optimal = req_to_instances(s)
    return -((a - a_opt) ** 2)


class Env:
    def __init__(self):
        self.s = None


    def reset(self):
        self.s = sample_states(1).squeeze()
        return self.s


    def step(self, a):
        r = reward(self.s, a)
        s2 = sample_states(1).squeeze()