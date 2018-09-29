
# imports
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from envs.spot_env import SpotEnv
from agents.random_agent import RandomAgent
from session_runner import SessionRunner


# The environment
env = SpotEnv()

# The policy and the agent
agent = RandomAgent(env.action_space)


# The agent

# evaluate
nr_episodes = 5 # Train for 5 episodes

runner = SessionRunner(env, agent)
performance = runner.run_session(nr_episodes)




# plot_performance
plt.figure()
plt.plot(performance.keys(), performance.values(), label='performance')
plt.xlabel('episode nr')
plt.ylabel('reward')
plt.legend()
plt.show()
