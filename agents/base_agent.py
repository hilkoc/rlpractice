""" Some base classes for reinforcment learning """

class RlAgent(object):
    """ An agent interacts with the environment.
    Observes a state, takes an action and receives a reward. """

    def act(self, state):
        """ Observe the given state and react to it. Decide on an action to take and return it."""
        raise NotImplementedError()

    def receive_reward(self, reward, terminal):
        """ Receive the reward for the last action taken and boolean indicating whether the last state is terminal.
        The agent can use this to learn and adapt its behavior accordingly."""
        raise NotImplementedError()

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass

    def total_reward(self):
        """ Returns total reward for the last episode. Used for measuring performance."""
        raise NotImplementedError()


import gym.spaces
import numpy as np

class DiscretePolicySpace(gym.spaces.Discrete):
    """ A finite set of policies """

    def __init__(self, policies):
        """ policies: a list of Policy objects. """
        self.policies = policies
        n = len(policies)
        super(DiscretePolicySpace, self).__init__(n)

    def sample(self):
        """ Returns an index, not a policy."""
        return np.random.randint(self.n)

    def contains(self, x):
        """ Expects x to be an index, not a policy. Checks if x is within range."""
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
                x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    @property
    def shape(self):
        return ()

    def __repr__(self):
        return "DiscretePolicySpace(%d)" % self.policies

    def __eq__(self, other):
        return self.policies == other.policies


