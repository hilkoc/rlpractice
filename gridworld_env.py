import gym
from gym.spaces import Discrete, Box
import numpy as np

class GridWorld(gym.core.Env):
    """ A grid world environment """
    metadata = {'render.modes': ['human']}

    MOVEMENTS = ['up', 'right', 'down', 'left']
    MOVE_VECS = [(0,-1), (1,0), (0,1), (-1,0)]
    CELL_VALUES = ['cliff', 'flat', 'agent', 'finish']

    def __init__(self):
        super(GridWorld, self).__init__()
        self.STATE_H = 5
        self.STATE_W = 10
        self.action_space = Discrete(len(self.MOVEMENTS))
        self.observation_space = Box(low=0, high=len(self.CELL_VALUES), shape=(self.STATE_H, self.STATE_W))
        self.start_x, self.start_y = 1, 3
        self.finish_x, self.finish_y = 8, 4
        self.world = np.empty((self.STATE_H, self.STATE_W))
        self.agent_x, self.agent_y = self.start_x, self.start_y

    def _step(self, action):
        assert self.action_space.contains(action)
        self.world[self.agent_y, self.agent_x] = self.CELL_VALUES.index('flat')
        dx, dy = self.MOVE_VECS[action]
        self.agent_y += dy
        self.agent_x += dx
        if 0 <= self.agent_y < self.STATE_H and 0 <= self.agent_x < self.STATE_W:
            reward = -1
            done = False
            if self.agent_y == self.finish_y and self.agent_x == self.finish_x:
                reward = 1
                done = True
            elif self.world[self.agent_y, self.agent_x] == self.CELL_VALUES.index('cliff'):
                reward = -100
                done = True
            self.world[self.agent_y][self.agent_x] = self.CELL_VALUES.index('agent')
        else:
            reward = -100
            done = True
            # raise IndexError("Invalid Move")
        obs = self.world
        return (obs, reward, done, {})

    def _reset(self):
        self.world.fill(self.CELL_VALUES.index('flat'))
        self.world[self.start_y][self.start_x] = self.CELL_VALUES.index('agent')
        self.world[self.finish_y][self.finish_x] = self.CELL_VALUES.index('finish')
        cliff_idx = self.CELL_VALUES.index('cliff')
        self.world[3, 2:8] = cliff_idx
        self.world[4, self.start_x + 1] = cliff_idx
        # Move agent back to the start
        self.agent_x, self.agent_y = self.start_x, self.start_y
        return self.world

    def _render(self, mode='human', close=False):
        """ Print the world to stdout"""
        print(self.world)

    def _close(self):
        pass

    def _seed(self, seed=None):
        seed = 0  # No randomness
        return [seed]

