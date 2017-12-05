from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from collections import deque
import keras
import gym
import random
import numpy as np
import pandas as pd
def create_model(state_dim, number_of_actions):
    model = Sequential()
    model.add(Dense(output_dim=16, activation='relu', input_dim=state_dim))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=number_of_actions, activation='linear'))
    opt = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=opt)
    return model

class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer=deque(maxlen=size)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def add_to_buffer(self, experience):
        self.buffer.append(experience)

def generator(number):
    return(i for i in range(number))

def epsilon_greedy_policy(q_values, epsilon):
    number_of_actions =len(q_values)
    action_probabilites = np.ones(number_of_actions, dtype=float)*epsilon/number_of_actions
    best_action = np.argmax(q_values)
    action_probabilites[best_action]+= (1-epsilon)
    return np.random.choice(number_of_actions, p=action_probabilites)

class DQNAgent:
    def __init__(self, env, model, gamma):
        self.env=env
        self.model=model
        self.replay_buffer=ReplayBuffer()
        self.gamma=gamma
        self.state_dim=env.observation_space.shape[0]

    def train_model(self, training_data, training_label):
        self.model.train_on_batch(training_data, training_label)

    def predict_one(self, state):
        return self.model.predict(state.reshape(1, self.state_dim)).flatten()

    def experience_replay(self, experiences):
        states, actions, rewards, next_states=zip(*[[experience[0], experience[1], experience[2], experience[3]] for experience in experiences])
        states=np.asarray(states)
        place_holder_state=np.zeros(self.state_dim)
        next_states_ = np.asarray([(place_holder_state if next_state is None else next_state) for next_state in next_states])
        q_values_for_states=self.model.predict(states)
        q_values_for_next_states=self.model.predict(next_states_)
        for x in generator(len(experiences)):
            y_true=rewards[x]
            if next_states[x] is not None:
                y_true +=self.gamma*(np.amax(q_values_for_next_states[x]))
            q_values_for_states[x][actions[x]]=y_true
        self.train_model(states, q_values_for_states)

    def fit(self, number_of_epsiodes, batch_size):
        for _ in generator(number_of_epsiodes):
            total_reward=0
            state=env.reset()
            while True:
                self.env.render()
                q_values_for_state=self.predict_one(state)
                action=epsilon_greedy_policy(q_values_for_state, 0.01)
                next_state, reward, done, _=env.step(action)
                if done:
                    next_state = None
                self.replay_buffer.add_to_buffer([state, action, reward, next_state])
                state = next_state
                total_reward += reward
                if len(self.replay_buffer.buffer) > 50:
                    experience=self.replay_buffer.sample(batch_size)
                    self.experience_replay(experience)
                if done:
                    break
            print("Total reward:", total_reward)


env = gym.make('CartPole-v0')
model=create_model(env.observation_space.shape[0], env.action_space.n)
agent=DQNAgent(env, model, 0.9)
agent.fit(100000, 32)
