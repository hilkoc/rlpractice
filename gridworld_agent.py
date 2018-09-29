from cmath import polar

import gym
import gym.spaces
class TestSpace(gym.spaces.Discrete):
    """ A finite set of policies """
    
print("het wrkt")
from agent import RlAgent, Environment, DiscretePolicySpace
import gridworld_env
import numpy as np

###########################
### Now gridworld specific
###########################

class FixedGridworldAgent(RlAgent):
    """ An agent for the gridworld with fixed behavior. Total steps 12, total reward -10"""

    def __init__(self):
        self.turn = 0
        self.actions = [0, 0] + [1] * 7 + [2] * 3
        self.total_steps = len(self.actions)
        self.total_reward = 0

    def _reset(self):
        print("Done, total reward %d" % self.total_reward)
        self.total_reward = 0

    def receive_state(self, state):
        """ The fixed agent always returns the same actions in the same order """
        print("Turn %d" % self.turn)
        action = self.actions[self.turn % self.total_steps]
        return action

    def receive_reward(self, reward, terminal):
        """ This agent has fixed behavior and so ignores awards. """
        self.turn += 1
        self.total_reward += reward
        if terminal:  # self.turn % self.total_steps == 0:
            self._reset()



# Action space is a set of policies.

class Policy(object):
    """ The inteface for every policy. When implementing a policy the _select_action method must be overridden """

    def _select_action(self, state):
        raise NotImplementedError

    def select_action(self, state):
        return self._select_action(state)


class ConstantPolicy(Policy):
    """ The constant policy always returns the same action. """
    def __init__(self, action):
        self.action = action

    def _select_action(self, state):
        # print('Constant policy returning: %d' % self.action)
        return self.action


class ExamplePolicy(Policy):
    """ The example policy returns actions made by another agent. """
    def __init__(self, rlAgent):
        self.agent = rlAgent

    def _select_action(self, state):
        a = self.agent._choose_action_idx(state)
        # print('Exmple policy returning: %d' % a)
        return self.agent.action_for_idx(a, state)


def make_constant_policy_space(discrete_action_space):
    """ Creates a discrete policy space of constant policies. """
    policies = [ConstantPolicy(a) for a in range(discrete_action_space.n)]
    policy_space = DiscretePolicySpace(policies)
    return policy_space


class TabularSarsa(RlAgent):
    """ An agent that learns from total reward accumulated over n-steps"""

    def __init__(self, n, observation_space, action_space, eps=0.1, gamma=0.9, alpha=0.3):
        self.n = n
        self._reset()
        self.observation_space = observation_space
        self.action_space = action_space  # Assuming discrete action space
        self.qvalues = np.zeros(observation_space.shape + tuple([action_space.n]))
        self.eps = eps
        self.gamma = gamma
        self.gamma_pow = np.power(self.gamma, range(self.n + 1))
        self.alpha = alpha
        # specific to this gridworld env
        self.my_value = gridworld_env.GridWorld.CELL_VALUES.index('agent')
        print("qvalues shape %s" % str(self.qvalues.shape))

    def _reset(self):
        self.states = []  # Instead of the whole state, we just store the current location here
        self.actions = []
        self.rewards = [0]
        self.turn = 0
        self.update_time = -self.n -1 # To update we need S_tau+n, which is available at tau +n + 1
        self.end_time = float('inf')  # infinity

    def receive_state(self, state):
        """ Instead of the whole state, we just store the current location """
        self.turn += 1
        self.update_time += 1
        # print("Turn %d" % self.turn)
        my_location = np.where(state == self.my_value)
        # print(my_location)
        assert my_location[0].shape == (1,)  # Check that we found something
        assert my_location[1].shape == (1,)
        self.states.append(my_location)
        action_idx = self._choose_action_idx(my_location)
        self.actions.append(action_idx)
        action = self.action_for_idx(action_idx, my_location)
        return action

    def _choose_action_idx(self, my_location):
        """ Epsilon greedy choose an action."""
        q_values = self.qvalues[my_location]
        # assert q_values.ndim == 1
        # print("choose action from shape %s" % str(q_values.shape))
        nb_actions = q_values.shape[1]
        assert nb_actions == self.action_space.n

        if np.random.uniform() < self.eps:
            action_idx = self.action_space.sample()  # np.random.random_integers(0, nb_actions-1)
        else:
            action_idx = np.argmax(q_values)  # The index of the action
        return action_idx

    def action_for_idx(self, action_idx, my_location):
        action = self.action_space.policies[action_idx].select_action(my_location)
        return action

    def receive_reward(self, reward, terminal):
        """ This agent has fixed behavior and so ignores awards. """
        self.rewards.append(reward)

        if terminal:
            print("Done, total reward %f" % sum(self.rewards))
            print("Path %s" % str(self.actions))
            # print("Last position %s:%s", self.states[-1])
            self.end_time = self.turn
            if self.update_time < 0:
                self.update_time = 0
            # continue learning the last rewards for T-n to T-1
            while self.update_time < self.end_time:
                self._learn_reward(reward)
                self.update_time += 1
            self._reset()
        elif self.update_time >= 0:
            self._learn_reward(reward)

    def _learn_reward(self, reward):
        """ Improve the qvalues learned using the given reward"""
        # G = sum_{i=0 to n-1}  gamma^i * R_i + gamma^n * qvalue(state, action at time t+n)
        # New qvalue = old qvalue  +  alpha * ( G - old qvalue)
        assert self.update_time >= 0

        discounted_rewards = 0.0
        for i in range(min(self.n, self.end_time - self.update_time)):
            discounted_rewards += self.gamma_pow[i] * self.rewards[self.update_time + 1 + i]

        G = discounted_rewards
        s, a = 0, 0
        # debug = (len(self.states), len(self.actions), len(self.rewards))
        t = self.update_time + self.n
        if t < self.end_time:
            s = self.states[t]
            a = self.actions[t]
            q_estimate = self.qvalues[s[0][0], s[1][0], a]
            G += self.gamma_pow[self.n] * q_estimate

        s = self.states[self.update_time]
        a = self.actions[self.update_time]  # a is the action_idx
        old_qvalue = self.qvalues[s[0][0], s[1][0], a]
        update = self.alpha * (G - old_qvalue)
        self.qvalues[s[0][0], s[1][0], a] += self.alpha * (G - self.qvalues[s[0][0], s[1][0], a])
        new_qvalue = self.qvalues[s[0][0], s[1][0], a]
        # print("state %s, action %s, old_q %s, new_q %s" % (s, a, old_qvalue, new_qvalue))


    def load_weights(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.qvalues = pickle.load(f)

    def save_weights(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.qvalues, f)

    def total_reward(self):
        return sum(self.rewards)


def run_a_session(n, alpha, gamma=0.9):
    env = gridworld_env.GridWorld()
    # agent = FixedGridworldAgent()
    policy_space = make_constant_policy_space(env.action_space)
    agent = TabularSarsa(n, env.observation_space, policy_space, eps=0.1, alpha=alpha, gamma=gamma)
    filename = 'qvalues_gridworld-n{}-a{}.np'.format(n, alpha)
    # agent.load_weights(filename)
    environment = Environment(env)
    environment.add_agent(agent)

    nr_episodes = 500 # Train for 500 episodes, then reduce alpha by half
    #average_reward = environment.run_session(nr_episodes)
    #agent.alpha /= 2.0
    average_reward = environment.run_session(nr_episodes)
    #agent.save_weights(filename)
    print("Average training reward for {} episodes: {}".format(nr_episodes, average_reward))
    nr_episodes = 10
    average_reward = environment.run_session(nr_episodes)
    print("Average value reward for {} episodes: {}".format(nr_episodes, average_reward))
    return average_reward


import matplotlib.pyplot as plt

def figure7_2():
    # all possible steps
    steps = [1,2,4,8,12]

    # all possible alphas
    alphas = np.arange(0.1, 0.7, 0.1)

    # perform 100 independent runs
    runs = 1

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in range(0, runs):
        for stepInd, step in zip(range(len(steps)), steps):
            for alphaInd, alpha in zip(range(len(alphas)), alphas):
                print('step:', step, 'alpha:', alpha)
                value = run_a_session(step, alpha)
                errors[stepInd, alphaInd] += value

    plt.figure()
    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('value')
    plt.legend()
    # plt.show()

# figure7_2()

def run_deterministic():
    """ Run with alpha=0, eps=0 """
    n=12
    alpha='0.2'
    filename = 'qvalues_gridworld-n{}-a{}.np'.format(n, alpha)
    env = gridworld_env.GridWorld()
    policy_space = make_constant_policy_space(env.action_space)
    agent = TabularSarsa(n, env.observation_space, policy_space, eps=0.0, alpha=0)
    agent.load_weights(filename)
    environment = Environment(env)
    environment.add_agent(agent)

    nr_episodes = 1
    average_reward = environment.run_session(nr_episodes)
    print('average_reward', average_reward, 'n', n, 'alpha', alpha)


def map_to_plot(map):
    sorted_keys = sorted(map.keys())
    xv, yv = sorted_keys, [map[k] for k in sorted_keys]
    return xv, yv

def get_performance(nr_episodes, n, alpha, gamma=0.9):
    env = gridworld_env.GridWorld()
    policy_space = make_constant_policy_space(env.action_space)
    agent = TabularSarsa(n, env.observation_space, policy_space, eps=0.1, alpha=alpha, gamma=gamma)
    # filename = 'qvalues_gridworld-n{}-a{}.np'.format(n, alpha)
    # agent.load_weights(filename)
    environment = Environment(env)
    environment.add_agent(agent)

    perf = environment.run_session(nr_episodes)
    # agent.save_weights(filename)
    time, best_reward = map_to_plot(perf)
    return time, best_reward


def learn_from_example(nr_episodes, n, alpha, gamma=0.9):
    env = gridworld_env.GridWorld()

    #make the example policy
    const_policy_space = make_constant_policy_space(env.action_space)
    # alpha, gamma don't matter here, because this agent never learns. Just an example.
    exampleAgent = TabularSarsa(n, env.observation_space, const_policy_space, eps=0, alpha=1, gamma=1)
    filename = 'qvalues_gridworld_example.np'
    exampleAgent.load_weights(filename)
    examplePolicy = ExamplePolicy(exampleAgent)

    policies = [ConstantPolicy(a) for a in range(env.action_space.n)]
    policies.insert(0, examplePolicy)
    policy_space = DiscretePolicySpace(policies)

    agent = TabularSarsa(n, env.observation_space, policy_space, eps=0.15, alpha=alpha, gamma=gamma)
    # filename = 'qvalues_gridworld_learn_by_example.np'
    # agent.load_weights(filename)
    environment = Environment(env)
    environment.add_agent(agent)

    perf = environment.run_session(nr_episodes)
    # agent.save_weights(filename)
    time, best_reward = map_to_plot(perf)
    return time, best_reward


def plot_performance():
    nr_episodes = 6

    # all possible steps
    steps = [10]

    # all possible alphas
    alphas = [0.15]  #np.arange(0.25, 0.35, 0.1)

    gammas = [0.99, 0.9, 0.8, 0.1]
    plt.figure()
    plt.plot([1,nr_episodes], [-8, -8], label='best possible')

    for n in steps:
        for alpha in alphas:
            for gamma in gammas:
                xv, yv = learn_from_example(nr_episodes, n, alpha, gamma)
                plt.plot(xv, yv, label='n : {}, alpha : {}, gamma {}'.format(n, alpha, gamma), marker='*')
    plt.xlabel('time')
    plt.ylabel('best reward')
    plt.legend()
    plt.show()


plot_performance()
