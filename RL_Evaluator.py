
# imports
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from envs.spot_env import SpotEnv
from agents.policy_gradient_agent import PolicyGradientAgent
from session_runner import SessionRunner


# The environment
env = SpotEnv()

# The policy and the agent
# agent = RandomAgent(env.action_space)
agent = PolicyGradientAgent(env.observation_space, env.action_space, alpha=0.040)


# The agent

# evaluate
nr_episodes = 1002

runner = SessionRunner(env, agent)
performance = runner.run_session(nr_episodes)




# plot_performance
plt.figure()
plt.plot(performance.keys(), performance.values(), label='performance')
plt.xlabel('episode nr')
plt.ylabel('reward')
plt.legend()
plt.show()
