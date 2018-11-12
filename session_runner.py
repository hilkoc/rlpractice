
class SessionRunner(object):
    """ Runs sessions and episodes. A session is a number of episodes.
    An episode is a number of steps, from reset untill done. """

    def __init__(self, env, agent, max_steps=5000):
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
        peek = nr_episodes//5 + 1
        for i in range(nr_episodes):

            state = env.reset()
            episode_reward = 0
            t = 0
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
                    raise RuntimeError("Max steps exceeded: {}".format(t))
            if i % peek == 0:
                print("\nEpisode %i" % i)
                print(agent.policy.theta)
                env.render()
                print(episode_reward)
            performance[i] = episode_reward

        return performance