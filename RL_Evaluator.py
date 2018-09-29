
# coding: utf-8

# # Evaluate Reinforcement Learning agents
# For a given environment and agent, run a number of episodes to evaluate the performance.



# imports
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', '')
# %matplotlib
import gridworld_env
import gridworld_agent
from agent import Environment




# The environment
env = gridworld_env.GridWorld()




# The policy and the agent
policy_space = gridworld_agent.make_constant_policy_space(env.action_space)
alpha, gamma= 1, 0.9
n = 5
agent = gridworld_agent.TabularSarsa(n, env.observation_space, policy_space, eps=0.1, alpha=alpha, gamma=gamma)
# filename = 'qvalues_gridworld-n{}-a{}.np'.format(n, alpha)
# agent.load_weights(filename)




# The agent




# evaluate
nr_episodes = 5 # Train for 5 episodes

environment = Environment(env)
environment.add_agent(agent)
performance = environment.run_session(nr_episodes)





# plot_performance
plt.figure()
plt.plot(performance.keys(), performance.values(), label='best possible')
plt.xlabel('time')
plt.ylabel('best reward')
plt.legend()
plt.show()

