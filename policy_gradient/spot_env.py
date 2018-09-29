import agent

print("agent imported from")
print(agent.__file__)

import gym
from gym.spaces import Discrete, Box
import numpy as np

class SpotEnv(gym.core.Env):
    """ Trade an asset with a fluctuating price """
    metadata = {'render.modes': ['human']}

    ACTIONS = ['buy', 'hold', 'sell']

    def __init__(self):
        super(SpotEnv, self).__init__()
        self.action_space = Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=-0, high=1000, shape=(1,))
        self.MAX_STEPS = 35
        self._reset()
        self.time = 0

    def _step(self, action):
        assert self.action_space.contains(action)
        done = self.time == self.MAX_STEPS:
        spot = self.vol * np.sin(self.time / 2.0) + 100
        self.time += 1
        if ACTIONS[action] == 'hold':
            reward = 0
        elif ACTIONS[action] == 'buy':
            self.asset += self.cash / spot * (1 -fee)
            self.cash = 0
        elif ACTIONS[action] == 'sell':
            self.cash += self.asset * spot * (1 -fee)
            self.asset = 0
        balance = self.current_value(spot)
        stepreturn = balance / self.prev_balance
        self.prev_balance = balance
        obs = self.spot
        reward = np.ln(stepreturn)  # Return log, so rewards are additive
        return (obs, reward, done, {})

    def _reset(self):
        self.start_spot = 100
        self.asset = 0.0
        self.cash = 100.0
        self.prev_balance = current_value(self.start_spot)
        self.fee = 0.0  # percentage
        self.vol = 8.0
        return self.start_spot

    def _render(self, mode='human', close=False):
        """ Print the world to stdout"""
        print("cash {:5d}  asset {:5d} balance {:5d}".format(self.cach, self.asset, self.current_value(100)))

    def _close(self):
        pass

    def _seed(self, seed=None):
        seed = 0  # No randomness
        return [seed]
        
    def current_value(self, spot):
        " Returns the cash value of the portfolio "
        return self.asset * spot + self.cash

