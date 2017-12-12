import random
import warnings
import numpy as np
from rl.util import *
from keras.optimizers import Adam, RMSprop
from keras import losses

class DQNAgent:
    def __init__(self, env, model, policy, replay_buffer, gamma=0.99, batch_size=64, tau=0, target_model_update_interval=10000, render=False):
        self.env=env
        self.trainable_model=model
        self.target_model=model
        self.policy=policy
        self.replay_buffer=replay_buffer
        self.gamma=gamma
        self.batch_size=batch_size
        self.tau=tau
        self.render=render
        self.target_model_update_interval=target_model_update_interval
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.n

        if self.render:
            warnings.warn("Warning: Rendering environment will make the training extremely slow")
        if self.tau>0:
            warnings.warn("Warning: Polyvak Averaging will be used to update target model at every step")

    def train_model(self, training_data, training_label, batch_size=64):
        self.trainable_model.fit(training_data, training_label, batch_size=self.batch_size, verbose=0) #Training the network

    def update_target_model(self):
        if self.tau>0:
            self.target_model=update_model_by_polyak_average(self.target_model, self.trainable_model, self.tau) #Updating the target network using the Polyak Averaging
        else:
            self.target_model=clone_weights(self.target_model, self.trainable_model) #Updating the target network in case Polyak Averaging is not used

    def select_action(self, state):
        q_values_for_state=self.compute_q_values(state) #Compute Q value for the current state
        return self.policy.select_action(q_values_for_state) #Select action based on the polic

    def compute_q_values(self, states, target=False):
        if (states.shape)==(self.state_dim,):
            return self.trainable_model.predict(states.reshape(1, self.state_dim)).flatten() #Predicting Q values of a single state
        if target:
            return self.target_model.predict(states) #Querying target network for Q values of multiple states
        return self.trainable_model.predict(states) #Querying the trainable network for Q values of multiple states

    def experience_replay(self):
        experiences = self.replay_buffer.sample(self.batch_size)
        states, next_states=zip(*[[experience[0], experience[3]] for experience in experiences]) #Seperating states, actions, rewards and next states
        states=np.asarray(states) #Converting to numpy array
        place_holder_state=np.zeros(self.state_dim)
        next_states_=np.asarray([(place_holder_state if next_state is None else next_state) for next_state in next_states]) #Converting to numpy array
        q_values_for_states=self.compute_q_values(states)
        q_values_for_next_states=self.compute_q_values(next_states_, True) #Computing the max Q(S',A') for the using the target network
        batch_len=len(experiences)
        training_data=np.zeros((batch_len, self.state_dim))
        training_label=np.zeros((batch_len, self.action_dim))
        index=0
        for state, action, reward, next_state in experiences:
            y_true = q_values_for_states[index]
            if next_state is None:
                y_true[action] = reward
            else:
                y_true[action] = reward + (self.gamma * np.amax(q_values_for_next_states[index]))
            training_data[index]=state
            training_label[index]=y_true
            index+=1
        self.train_model(training_data, training_label)


    def compile(self, opt=RMSprop(lr=0.00025), loss):
        self.target_model=clone_model(self.trainable_model)
        self.trainable_model.compile(loss=loss, optimizer=opt)
        self.target_model.compile(loss=loss, optimizer=opt)


    def fit(self, number_of_epsiodes):
        for episode in range(number_of_epsiodes): #Looping through total number of episodes
            total_reward=0 #The total reward an agent will get after an epsiode
            state=self.env.reset() #Resetting environment
            while True:
                self.env.render() if self.render == True else False #Rendering the environment
                action=self.select_action(state) #Select action based on the policy
                next_state, reward, done, _=self.env.step(action)
                if done:
                    next_state=None #If the next state is terminal, mark it as None

                self.replay_buffer.add_to_buffer((state, action, reward, next_state)) #Adding experience to replay buffer
                self.experience_replay() #Experience replay step

                state=next_state
                total_reward +=reward
                if done:
                    break
            if self.tau>0 or episode%self.target_model_update_interval==0:
                self.update_target_model()
            print("Total reward ", total_reward)
