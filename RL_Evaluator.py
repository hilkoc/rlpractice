
# imports
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from envs.spot_env import SpotEnv
from agents.policy_gradient_agent import PolicyGradientAgent
from agents.spot_agent import SpotEnvAgent
from session_runner import SessionRunner


# The environment
env = SpotEnv()

# The policy and the agent
# agent = RandomAgent(env.action_space)
agent = PolicyGradientAgent(env.observation_space, env.action_space, alpha=0.01)
#benchmark_agent = SpotEnvAgent(env)

# The benchmark agent
# Episode 1
# [ 91.95142244 108.04857756]  # cos(0.75)
# cash   0.00  asset   7.77 balance  776.90
# 19.65338405623287


# evaluate
nr_episodes = 400

runner = SessionRunner(env, agent)
performance = runner.run_session(nr_episodes)




def plot_performance():
    plt.figure()
    plt.plot(performance.keys(), performance.values(), label='performance')
    plt.xlabel('episode nr')
    plt.ylabel('reward')
    plt.legend()
    plt.show()

import pandas as pd

def plot_pi():
    theta = agent.policy.theta
    # agent.policy.theta[2] = 0.9
    print(theta)

    s_axis = np.arange(88, 112, 0.5)
    pi_func = np.array([agent.policy.policy_func(state) for state in s_axis])
    # h1 = pi_func[:, 1]

    data = {'h' + str(i) : pi_func[:,i] for i in range(3)}
    data['total'] = data['h0'] + data['h1'] + data['h2'] - 0.2
    df = pd.DataFrame(data, index=s_axis)
    df.plot()

plot_pi()
plot_performance()