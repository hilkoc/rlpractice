""" Some base classes for reinforcment learning """

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

    def total_reward(self):
        """ Returns total reward for the last episode. Used for measuring performance."""
        raise NotImplementedError()


class Environment(object):
    """ Runs sessions and episodes. A session is a number of episodes.
    An episode is a number of steps, from reset untill done """

    def __init__(self, env):
        self.env = env
        self.agent = None
        self.max_turns_per_episode = 50

    def add_agent(self, agent):
        # for now just a single agent environment
        self.agent = agent

    def run_session(self, nr_episodes=1):
        """
        :param nr_episodes: integer
        :return: A map showing total reward for best episodes during this session
        """
        performance = dict()
        best = -float('inf')
        env, agent = self.env, self.agent
        for i in range(nr_episodes):
            # LOG(INFO) << "Starting episode " << episode_nr;
            print("\nEpisode %i" % i)
            # Stats episode_stats;
            t = 0
            episode_reward =0
            state = env.reset()
            done = False
            while not done:
                action = agent.receive_state(state)
                state, reward, done, _ = env.step(action)
                # print("Action: %s Reward: %s" % (str(action),str(reward)))
                agent.receive_reward(reward, done)
                t += 1
                episode_reward += reward
                if t >= self.max_turns_per_episode:
                    done = True
            # this->finalize_episode(episode_nr, episode_stats, session_stats);
            if episode_reward >= best:
                best = episode_reward
                performance[i] = best
        # session_stats.log_summary();
        return performance


import gym.spaces
import numpy as np

class DiscretePolicySpace(gym.spaces.Discrete):
    """ A finite set of policies """

    def __init__(self, policies):
        """ policies: a list of Policy objects. """
        self.policies = policies
        n = len(policies)
        super(DiscretePolicySpace, self).__init__(n)

    def sample(self):
        """ Returns an index, not a policy."""
        return np.random.randint(self.n)

    def contains(self, x):
        """ Expects x to be an index, not a policy. Checks if x is within range."""
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
                x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    @property
    def shape(self):
        return ()

    def __repr__(self):
        return "DiscretePolicySpace(%d)" % self.policies

    def __eq__(self, other):
        return self.policies == other.policies


