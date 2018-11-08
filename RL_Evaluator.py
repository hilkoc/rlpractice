
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
agent = PolicyGradientAgent(env.observation_space, env.action_space, alpha=0.40)
#benchmark_agent = SpotEnvAgent(env)

# The benchmark agent
# Episode 1
# [ 91.95142244 108.04857756]  # cos(0.75)
# cash   0.00  asset   7.77 balance  776.90
# 19.65338405623287


# evaluate
nr_episodes = 502

runner = SessionRunner(env, agent)
performance = runner.run_session(nr_episodes)




# plot_performance
plt.figure()
plt.plot(performance.keys(), performance.values(), label='performance')
plt.xlabel('episode nr')
plt.ylabel('reward')
plt.legend()
plt.show()
