from agent import Agent

class DDPG(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)

    def select_action(self, state):
        return super().select_action(state)
