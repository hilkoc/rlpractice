import agents.base_agent

class RandomAgent(agents.base_agent.RlAgent):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()

    def receive_reward(self, reward, terminal):
        pass # must be implmented

    def total_reward(self):
        pass # must be implmented