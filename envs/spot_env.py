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
        self.observation_space = Box(low=0, high=1000, shape=(1,))
        self.MAX_STEPS = 35
        self.time = 0
        self._reset()

    def _step(self, action):
        assert self.action_space.contains(action)
        done = self.time == self.MAX_STEPS
        spot = self.vol * np.sin(self.time / 2.0) + 100
        self.time += 1
        if self.ACTIONS[action] == 'hold':
            reward = 0
        elif self.ACTIONS[action] == 'buy':
            self.asset += self.cash / spot * (1 - self.fee)
            self.cash = 0
        elif self.ACTIONS[action] == 'sell':
            self.cash += self.asset * spot * (1 - self.fee)
            self.asset = 0
        balance = self.current_value(spot)
        stepreturn = balance / self.prev_balance
        self.prev_balance = balance
        obs = spot
        reward = 100 * np.log(stepreturn)  # Return log, so rewards are additive
        return obs, reward, done, {}

    def _reset(self):
        print("Reset at time was %i" % self.time)
        self.time = 0
        self.start_spot = 100
        self.asset = 0.0
        self.cash = 100.0
        self.prev_balance = self.current_value(self.start_spot)
        self.fee = 0.0  # percentage
        self.vol = 8.0
        return self.start_spot

    def _render(self, mode='human', close=False):
        """ Print to stdout"""
        print("cash {: 6.2f}  asset {: 6.2f} balance {: 6.2f}".format(self.cash, self.asset, self.current_value(100)))

    def _close(self):
        pass

    def _seed(self, seed=None):
        seed = 0  # No randomness
        return [seed]
        
    def current_value(self, spot):
        " Returns the cash value of the portfolio "
        return self.asset * spot + self.cash

