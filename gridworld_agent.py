import gridworld_env

env = gridworld_env.GridWorld()

done = False
turn = total_reward = 0
obs = env.reset()
obs, reward, done, _ = env.step(0)
turn += 1
while not done:
    action = 1
    if turn >= 8:
        action = 2
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    turn += 1
    print(obs)
    print(reward)
print("Done, total reward %d" % total_reward)