import crafter
import gym

env = gym.make('CrafterReward-v1')
print(env.observation_space.shape)