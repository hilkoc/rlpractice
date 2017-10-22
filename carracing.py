# for i in range(episode_count):
#     done, reward = False, 0
#     ob = env.reset()
#     log.info("Episode %d", i)
#     while not done:
#         action = agent.act(ob, reward, done)
#         log.debug("Action: %s", action)
#         ob, reward, done, _ = env.step(action)
#         # log.info(ob, reward, done)
#         env.render()

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from sarsa_nstep import SarsaNStepAgent
from rl.policy import EpsGreedyQPolicy
from rl.core import Processor


ENV_NAME = 'CarRacing-v0'  # Has action space Box(3,)
#ENV_NAME = 'CartPole-v0'  # Has action space Discrete(2)
#ENV_NAME = 'Pendulum-v0'  # Has action space Box(1,)

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(1234)
env.seed(1234)

class CarRacingProcessor(Processor):
    """ We convert a discrete action into a continues one """

    def __init__(self):
        self.actions = list()
        self.actions.append(np.array([0.,1.,0.]))  # full speed
        self.actions.append(np.array([0., 0.5, 0.]))  # half speed
        self.actions.append(np.array([-0.7, 0., 0.]))  # steer
        self.actions.append(np.array([0.7, 0., 0.]))  # steer
        self.actions.append(np.array([0., 0., 0.85]))  # brake

    def process_action(self, raw_action):
        """Processes an action predicted by an agent but before execution in an environment. """
        # print("raw_action is")
        # print(raw_action)
        return self.actions[raw_action]

processor = CarRacingProcessor()

nb_actions = len(processor.actions)
print("nb_actions %s" % nb_actions)
print("imput dim %s" % str(env.observation_space.shape))

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


policy = EpsGreedyQPolicy(eps=0.2)
sarsa = SarsaNStepAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=2, policy=policy, processor=processor)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# print("vout")
# batch_in = np.random.random_sample( (1,1) + env.observation_space.shape)
# v_out = model.predict(batch_in, batch_size=1, verbose=2)
# print(v_out)
# print(v_out.shape)

sarsa.load_weights('sarsa_nstep_{}_weights.h5f'.format(ENV_NAME))
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env, nb_steps=50, action_repetition=5, visualize=False, verbose=2)

# After training is done, we save the final weights.
# sarsa.save_weights('sarsa_nstep_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=2, visualize=True)
