""" Some base classes for reinforcement learning """

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

