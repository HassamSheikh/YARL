import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from rl.agents.DQN import DQNAgent
from rl.memory.ReplayBuffer import ReplayBuffer
from rl.policy.Policy import EpsilonGreedyPolicy
import gym

def create_model(state_dim, number_of_actions):
    model = Sequential()
    model.add(Dense(output_dim=16, activation='relu', input_dim=state_dim))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=number_of_actions, activation='linear'))
    return model

env = gym.make('CartPole-v0')
model=create_model(env.observation_space.shape[0], env.action_space.n)
replay_buffer=ReplayBuffer()
policy=EpsilonGreedyPolicy()
agent=DQNAgent(env, model, 0.1, replay_buffer, policy, 0.99)
agent.compile()
agent.fit(100000, 32)
