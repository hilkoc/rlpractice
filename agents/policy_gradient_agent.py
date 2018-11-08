import agents.base_agent
import numpy as np
np.random.seed(0)

class BasePolicy(object):
    """ The inteface for every policy."""

    def select_action(self, state):
        raise NotImplementedError


class LinearSoftmaxPolicy(object):
    """ The inteface for every policy."""

    def __init__(self, theta0, theta1):
        self.theta = np.array([theta0, theta1, 5.0], 'float64')

    def policy_func(self, state):
        th2 = self.theta[2]
        h0 = th2 * (self.theta[0] - state)
        h1 = 0.0  #001 * th2 * (1 - self.theta[0] + self.theta[1])
        h2 = th2 * (-self.theta[1] + state)
        h = np.array([h0,h1,h2])
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
        th2 = self.theta[2]
        grad_h0 = np.array([1 * th2, 0, self.theta[0] - state])
        grad_h1 = np.array([0.0, 0.0, 0.0])
        grad_h2 = np.array([0, -1 * th2, -self.theta[1] + state])
        g = self.policy_func(state)
        sum_grad_h_g = grad_h0 * g[0] + grad_h1 * g[1] + grad_h2 * g[2]
        r = [grad_h0 - sum_grad_h_g,
             grad_h1 - sum_grad_h_g,
             grad_h2 - sum_grad_h_g]
        return r


class PolicyGradientAgent(agents.base_agent.RlAgent):
    """ An agent interacts with the environment.
    Observes a state, takes an action and receives a reward. """

    def __init__(self, observation_space, action_space, gamma=0.9, alpha=0.1):
        self.observation_space = observation_space
        self.action_space = action_space  # Assuming discrete action space
        self.gamma = gamma
        # self.gamma_pow = np.power(self.gamma, range(self.n + 1))
        self.alpha = alpha
        self.policy = LinearSoftmaxPolicy(91.95142244, 108.04857756)
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
        # print("Received reward {}".format(reward))
        if terminal:
            G = 0
            T = len(self.rewards)
            # print("Terminal! {} and T = {}".format(terminal, T))
            for t in reversed(range(T)):
                G = self.gamma * G + self.rewards[t]
                action_t = self.actions[t]
                theta_update = self.alpha * (G - 0.0) * self.policy.grad_log(self.states[t])[action_t]
                # print("G = {}. Updating {} with {}".format(G, str(self.policy.theta), str(theta_update)))
                self.policy.theta += theta_update
                # For debugging
                if 80 < theta_update[1]:
                    raise "Theta exploding: G {} s {} a {}".format(G, self.states[t], self.actions[t])
            # Reset 3for the next episode
            self.reset()

    def base_line(self):
        return 15.0

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass

    def total_reward(self):
        """ Returns total reward for the last episode. Used for measuring performance."""
        return sum(self.rewards)
