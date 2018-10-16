import gym



env = gym.make('CartPole-v0')

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

print(STATE_DIM)
print(ACTION_DIM)

for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break