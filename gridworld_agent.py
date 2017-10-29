import gridworld_env


class RlAgent(object):
    """ An agent interacts with the environment.
    Observes a state, takes an action and receives a reward. """

    def receive_state(self, state):
        """ Observe the given state and react to it. Decide on an action to take and return it."""
        raise NotImplementedError()

    def receive_reward(self, reward, terminal):
        """ Receive the reward for the last action taken and boolean indicating whether the last state is terminal.
        The agent can use this to learn and adapt its behavior accordingly."""
        raise NotImplementedError()

    def load_weights(self, filename):
        pass

    def save_weights(self, filename):
        pass


class Environment(object):
    """ Runs sessions and episodes. A session is a number of episodes.
    An episode is a number of steps, from reset untill done """

    def __init__(self, env):
        self.env = env
        self.agent = None

    def add_agent(self, agent):
        # for now just a single agent environment
        self.agent = agent

    def run_session(self, nr_episodes=1):
        env, agent = self.env, self.agent
        for i in range(nr_episodes):
            # LOG(INFO) << "Starting episode " << episode_nr;
            # Stats episode_stats;
            t = 0
            state = env.reset()
            done = False
            while not done:
                if t%11 ==10:
                    print("Rendering State:")
                    env.render()
                action = agent.receive_state(state)
                state, reward, done, _ = env.step(action)
                # print("Action: %s Reward: %s" % (str(action),str(reward)))
                agent.receive_reward(reward, done)
                t += 1
            # this->finalize_episode(episode_nr, episode_stats, session_stats);
        # session_stats.log_summary();


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


import numpy as np
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
        print("qvalues shape %s"  % str(self.qvalues.shape))

    def _reset(self):
        self.states = []  #Instead of the whole state, we just store the current location here
        self.actions = []
        self.rewards = [0]
        self.turn = -1  # Start at -1 because we begin with increment
        self.update_time = -1 -self.n
        self.end_time = float('inf')  #infinity

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
        action = self._choose_action(my_location)
        self.actions.append(action)
        return action

    def receive_reward(self, reward, terminal):
        """ This agent has fixed behavior and so ignores awards. """
        self.rewards.append(reward)

        if terminal:
            print("Done, total reward %f" % sum(self.rewards))
            print("Path %s" % str(self.actions))
            print("Last position %s:%s", self.states[-1])
            print()
            self.end_time = self.turn
            if self.update_time < 0:
                self.update_time = 0
            # continue learning the last rewards for T-n to T-1
            while self.update_time < self.end_time:
                self._learn_reward(reward)
                self.update_time +=1
            self._reset()
        elif self.update_time >= 0:
            self._learn_reward(reward)

    def _choose_action(self, my_location):
        """ Epsilon greedy choose an action."""
        q_values = self.qvalues[my_location]
        # assert q_values.ndim == 1
        # print("choose action from shape %s" % str(q_values.shape))
        nb_actions = q_values.shape[1]
        assert nb_actions == self.action_space.n

        if np.random.uniform() < self.eps:
            action = self.action_space.sample()  # np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)  # The index of the action equals the action itself

        # Debug fix actions
        # acts = [0, 2, 3, 2, 2]
        # action = acts[self.turn]
        return action

    def _learn_reward(self, reward):
        """ Improve the qvalues learned using the given reward"""
        # G = sum_{i=0 to n-1}  gamma^i * R_i + gamma^n * qvalue(state, action at time t+n)
        # New qvalue = old qvalue  +  alpha * ( G - old qvalue)
        assert self.update_time >=0

        discounted_rewards = 0.0
        for i in range(min(self.n, self.end_time - self.update_time)):
            discounted_rewards += self.gamma_pow[i] * self.rewards[self.update_time + 1 + i]

        G = discounted_rewards
        s, a = 0, 0
        if self.update_time + self.n < self.end_time:
            s = self.states[self.turn]
            a = self.actions[self.turn]
            q_estimate = self.qvalues[s, a]
            G += self.gamma_pow[self.n] * q_estimate

        s = self.states[self.update_time]
        a = self.actions[self.update_time]
        self.qvalues[s, a] += self.alpha * (G - self.qvalues[s, a])

    def load_weights(self, filename):
        import pickle
        with open(filename,'rb') as f:
            self.qvalues = pickle.load(f)

    def save_weights(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.qvalues, f)


env = gridworld_env.GridWorld()
# agent = FixedGridworldAgent()
agent = TabularSarsa(12, env.observation_space, env.action_space, eps=0.1)
filename = 'qvalues_gridworld.np'
agent.load_weights(filename)
environment = Environment(env)
environment.add_agent(agent)

environment.run_session(5000)
agent.save_weights(filename)
print("qvalues")
print(agent.qvalues)