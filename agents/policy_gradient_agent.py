import agents.base_agent
import numpy as np
np.random.seed(0)

class BasePolicy(object):
    """ The inteface for every policy."""

    def select_action(self, state):
        raise NotImplementedError

    def grad_log(self, state):
        raise NotImplementedError



class LinearSoftmaxPolicy(BasePolicy):
    """ The inteface for every policy."""

    def __init__(self, theta0, theta1):
        self.theta = np.array([theta0, theta1, 1.0], 'float64')

    def policy_func(self, state):
        th2 = self.theta[2]  # temperature
        h0 = th2 * (self.theta[0] - state)
        h1 = 0.0  # th2 * (1 - self.theta[0] + self.theta[1])
        h2 = th2 * (-self.theta[1] + state)
        h = np.array([h0, h1, h2])
        f = np.exp(h)
        if not np.isfinite(f).all():
            print(h)
            print(f)
            raise ValueError("Theta exploded")
        denominator = sum(f)
        g = f / denominator
        return g

    def select_action(self, state):
        g = self.policy_func(state)
        # return np.argmax(g)
        eps = np.random.random_sample() #Uniform in [0,1)
        if eps < g[0]:
            return 0
        if eps < g[0] + g[1]:
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
        self.policy = LinearSoftmaxPolicy(99.0, 101.0)
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

        if terminal:
            G = 0
            T = len(self.rewards)

            for t in reversed(range(T)):
                G = self.gamma * G + self.rewards[t]
                action_t = self.actions[t]
                theta_update = self.alpha * (G - self.base_line()) * self.policy.grad_log(self.states[t])[action_t]
                self.policy.theta += theta_update
                # For debugging
                if 20 < abs(self.policy.theta[2]):
                    print("G = {}. Updating {} with {}".format(G, str(self.policy.theta), str(theta_update)))
                    raise ValueError("Theta exploding: G {} s {} a {}".format(G, self.states[t], self.actions[t]))
            # Reset 3for the next episode
            self.reset()

    def base_line(self):
        return 0.0

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass

    def total_reward(self):
        """ Returns total reward for the last episode. Used for measuring performance."""
        return sum(self.rewards)
