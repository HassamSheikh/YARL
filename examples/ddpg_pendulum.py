import sys
sys.path.append('../')
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Add, merge
from keras.optimizers import RMSprop, Adam
from rl.memory.ReplayBuffer import ReplayBuffer
from rl.policy.Policy import DecayingEpsilonGreedyPolicy
from rl.agents.DDPG import DDPGAgent
from rl.util import *
import gym

def create_critic_network(state_size, action_dim):
    state_input_layer = Input(shape=(state_size, ), name='state_input')
    action_input_layer = Input(shape=(action_dim, ),name='action_input')
    w1 = Dense(10, activation='relu')(state_input_layer)
    h1 = Dense(10, activation='linear')(w1)
    a1 = Dense(10, activation='linear')(action_input_layer)
    h2 = Add()([h1,a1])
    h3 = Dense(20, activation='relu')(h2)
    output_layer = Dense(action_dim, activation='linear', name='output_layer')(h3)
    model = Model(inputs=[state_input_layer, action_input_layer], outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def create_actor_network(self, state_size,action_dim):
    S = Input(shape=[state_size])
    h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
    h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
    Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
    Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
    Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
    V = merge([Steering,Acceleration,Brake],mode='concat')
    model = Model(input=S,output=V)
    return model, model.trainable_weights, S



env, state_dim, action_dim = build_gym_environment('Pendulum-v0')
critic = create_critic_network(state_dim, action_dim)

# model=create_model(env.observation_space.shape[0], env.action_space.n)
# replay_buffer=ReplayBuffer()
# policy=DecayingEpsilonGreedyPolicy()
# agent=DQNAgent(env, model, policy, replay_buffer, tau=0.99)
# agent.compile(RMSprop(lr=0.00025), losses.logcosh)
# agent.fit(10000)
