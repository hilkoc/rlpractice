import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from agents.policy_gradient_agent import PolicyGradientAgent
from envs.spot_env import SpotEnv
from agents.spot_agent import SpotEnvAgent
from session_runner import SessionRunner



env = SpotEnv()

agent = PolicyGradientAgent(env.observation_space, env.action_space, alpha=0.05)

agent = SpotEnvAgent(env)


nr_episodes = 1

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
    print(agent.policy.theta)

    s_axis = np.arange(88, 112, 0.1)
    pi_func = np.array([agent.policy.policy_func(state) for state in s_axis])

    data = {'h' + str(i) : pi_func[:,i] for i in range(3)}
    data['total'] = data['h0'] + data['h1'] + data['h2']
    df = pd.DataFrame(data, index=s_axis)
    df.plot()



plot_pi()
plot_performance()


"""
After training 1000 episodes, with MAX_STEPS=6283 and alpha=0.05 we find 
total reward: 946.498510405275
theta:        [ 91.58443278 107.65802878   2.30397928]

For comparison, the benchmark agent
total reward: 945.4199924697053
theta:       [ 91.22417438 108.77582562]

total reward: 975.8279175020341
theta:       [ 92.68311131 107.31688869]
"""