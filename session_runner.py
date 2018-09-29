
class SessionRunner(object):
    """ Runs sessions and episodes. A session is a number of episodes.
    An episode is a number of steps, from reset untill done. """

    def __init__(self, env, agent, max_steps=50):
        self.env = env
        self.agent = agent
        self.max_turns_per_episode = max_steps


    def run_session(self, nr_episodes=1):
        """
        :param nr_episodes: integer
        :return: A map showing the reward for each episode during this session
        """
        performance = dict()
        env, agent = self.env, self.agent
        for i in range(nr_episodes):
            print("\nEpisode %i" % i)

            t = 0
            episode_reward =0
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                # print("Action: %s Reward: %s" % (str(action),str(reward)))
                agent.receive_reward(reward, done)
                t += 1
                episode_reward += reward
                if t >= self.max_turns_per_episode:
                    done = True
            performance[i] = episode_reward

        return performance