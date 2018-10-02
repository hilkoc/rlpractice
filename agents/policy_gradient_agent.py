import agents.base_agent
import numpy as np


class BasePolicy(object):
    """ The inteface for every policy."""

    def select_action(self, state):
        raise NotImplementedError


class LinearSoftmaxPolicy(object):
    """ The inteface for every policy."""

    def __init__(self, theta1, theta2):
        self.theta = np.array([theta1, theta2], 'float64')

    def policy_func(self, state):
        h1 = self.theta[0] - state
        h2 = 0
        h3 = -self.theta[1] + state
        h = np.array([h1,h2,h3])
        f = np.exp(h)
        denominator = sum(f)
        g = f / denominator
        return g

    def select_action(self, state):
        eps = np.random.random_sample() #Uniform in [0,1)
        g = self.policy_func(state)
        a1, a2 = g[0], g[0] + g[1]
        if eps < a1:
            return 0
        if eps < a2:
            return 1
        return 2

    def grad_log(self, state):
        """ The gradient of the log of the policy_func,
            Returns a vector with len(theta) coordinates
            in a vector with coordinates for each action idx. """
        grad_h1 = np.array([1, 0])
        grad_h2 = np.array([0, 0])
        grad_h3 = np.array([0, -1])
        g = self.policy_func(state)
        sum_grad_h_g = grad_h1 * g[0] + grad_h2 * g[1] + grad_h3 * g[2]
        r = [grad_h1 - sum_grad_h_g,
             grad_h2 - sum_grad_h_g,
             grad_h3 - sum_grad_h_g]
        return r


class PolicyGradientAgent(agents.base_agent.RlAgent):
    """ An agent interacts with the environment.
    Observes a state, takes an action and receives a reward. """

    def __init__(self, observation_space, action_space, gamma=0.9, alpha=0.3):
        self.observation_space = observation_space
        self.action_space = action_space  # Assuming discrete action space
        self.gamma = gamma
        # self.gamma_pow = np.power(self.gamma, range(self.n + 1))
        self.alpha = alpha
        self.policy = LinearSoftmaxPolicy(98, 101.1)
        self.reset()

    def reset(self):
        self.rewards = list()
        self.states = list()
        self.actions = list()

    def act(self, state):
        """ Observe the given state and react to it. Returns an action from the action space."""
        action_idx = self.policy.select_action(state)
        self.states.append(state)
        self.actions.append(action_idx)
        return action_idx

    def receive_reward(self, reward, terminal):
        """ Receive the reward for the last action taken and boolean indicating whether the last state is terminal.
        The agent can use this to learn and adapt its behavior accordingly."""
        self.rewards.append(reward)
        print("Received reward {}".format(reward))
        if terminal:
            G = 0
            T = len(self.rewards)
            print("Terminal! {} and T = {}".format(terminal, T))
            for t in reversed(range(T)):
                G = self.gamma * G + self.rewards[t]
                action_t = self.actions[t]
                theta_update = self.alpha * G * self.policy.grad_log(self.states[t])[action_t]
                print("Updating {} with {}".format(str(self.policy.theta), str(theta_update)))
                self.policy.theta += theta_update
                # For debugging
                if 80 < theta_update[1]:
                    raise "Theta exploding: G {} s {} a {}".format(G, self.states[t], self.actions[t])
            # Reset for the next episode
            self.reset()

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass

    def total_reward(self):
        """ Returns total reward for the last episode. Used for measuring performance."""
        return sum(self.rewards)
