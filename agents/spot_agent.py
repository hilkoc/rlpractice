import agents.base_agent
import numpy as np
from envs.spot_env import SpotEnv

class Dummy(object):
    pass

class SpotEnvAgent(agents.base_agent.RlAgent):
    """ A deterministic agent for benchmarking. """

    def __init__(self, spot_env):
        avg, amp = spot_env.start_spot, spot_env.vol * np.cos(0.75)
        theta0 = avg - amp
        theta1 = avg + amp
        self.policy = Dummy()
        self.policy.theta = np.array([theta0, theta1, 1.0])
        self.reset()

    def reset(self):
        self.rewards = list()

    def act(self, state):
        """ Observe the given state and react to it. Decide on an action to take and return it."""
        if state < self.theta[0]:
            return 0
        if state > self.theta[1]:
            return 2
        return 1  # hold

    def receive_reward(self, reward, terminal):
        self.rewards.append(reward)

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass

    def total_reward(self):
        """ Returns total reward for the last episode. Used for measuring performance."""
        return sum(self.rewards)
