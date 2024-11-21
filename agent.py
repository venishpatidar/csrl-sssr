class Agent(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.num_inputs = num_inputs
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

    def select_action(self, state):
        pass

    def update_parameters(self, memory, updates):
        pass

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass

