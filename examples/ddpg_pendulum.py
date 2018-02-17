import sys
sys.path.append('../')
from keras.models import Model
from keras.layers import Dense, Activation, Input, Add
from keras.optimizers import RMSprop, Adam
from rl.memory.ReplayBuffer import ReplayBuffer
from rl.agents.DDPG import DDPGAgent
from rl.util import *
import gym
import numpy as np

def create_critic_network(state_size, action_dim):
    state_input_layer = Input(shape=(state_size,), name='critic_state_input')
    action_input_layer = Input(shape=(action_dim,),name='critic_action_input')
    w1 = Dense(10, activation='relu')(state_input_layer)
    h1 = Dense(10, activation='linear')(w1)
    a1 = Dense(10, activation='linear')(action_input_layer)
    h2 = Add()([h1,a1])
    h3 = Dense(20, activation='relu')(h2)
    output_layer = Dense(action_dim, activation='linear', name='critic_output_layer')(h3)
    model = Model(inputs=[state_input_layer, action_input_layer], outputs=output_layer)
    return model

def create_actor_network(state_size, action_dim):
    state_input_layer = Input(shape=(state_size,), name='actor_state_input')
    h0 = Dense(50, activation='relu')(state_input_layer)
    h1 = Dense(10, activation='relu')(h0)
    output_layer = Dense(action_dim, activation='linear', name='actor_output_layer')(h1)
    model = Model(inputs=state_input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='mse')
    return model



env, state_dim, action_dim = build_gym_environment('Pendulum-v0')
critic = create_critic_network(state_dim, action_dim)
actor = create_actor_network(state_dim, action_dim)
replay_buffer=ReplayBuffer()
random_process=OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=0.3)
agent=DDPGAgent(env, actor, critic, replay_buffer, random_process, tau=0.999)
agent.compile()
agent.fit(10000)
