import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from rl.agents.DQN import DQNAgent
from rl.memory.ReplayBuffer import ReplayBuffer
from rl.policy.Policy import DecayingEpsilonGreedyPolicy
import gym

def create_model(state_dim, number_of_actions):
    model = Sequential()
    model.add(Dense(output_dim=64, activation='relu', input_dim=state_dim))
    model.add(Dense(output_dim=number_of_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)
    return model

env = gym.make('CartPole-v0')
model=create_model(4, env.action_space.n)
replay_buffer=ReplayBuffer()
policy=DecayingEpsilonGreedyPolicy()
agent=DQNAgent(env, model, 0.99, replay_buffer, policy, 0)
agent.compile()
agent.fit(1000, 64)
