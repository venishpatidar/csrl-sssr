import torch

class Agent(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.num_inputs = num_inputs
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        # for reward normalization
        self.momentum = args.momentum
        self.mean = 0.0
        self.var = 1.0

        # Torch device
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
    def reward_normalization(self, rewards):
        # update mean and var for reward normalization
        batch_mean = torch.mean(rewards)
        batch_var = torch.var(rewards)
        self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
        std = torch.sqrt(self.var)
        normalized_rewards = (rewards - self.mean) / (std + 1e-8)
        return normalized_rewards

    def select_action(self, state):
        pass

    def update_parameters(self, memory, updates):
        pass

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass

