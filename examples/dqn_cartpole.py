import sys
sys.path.append('../')
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras import losses
from rl.agents.DQN import DQNAgent
from rl.memory.ReplayBuffer import ReplayBuffer
from rl.policy.Policy import DecayingEpsilonGreedyPolicy
import gym

def create_model(state_dim, number_of_actions):
    model = Sequential()
    model.add(Dense(output_dim=64, activation='relu', input_dim=state_dim))
    model.add(Dense(output_dim=number_of_actions, activation='linear'))
    return model

env = gym.make('CartPole-v0')
model=create_model(env.observation_space.shape[0], env.action_space.n)
replay_buffer=ReplayBuffer()
policy=DecayingEpsilonGreedyPolicy()
agent=DQNAgent(env, model, policy, replay_buffer, tau=0.99)
agent.compile(RMSprop(lr=0.00025), losses.logcosh)
agent.fit(10000)
