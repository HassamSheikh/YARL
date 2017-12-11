import random
import numpy as np
from rl.util import *
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, env, model, gamma, replay_buffer, policy, tau):
        self.env=env
        self.trainable_model=model
        self.target_model=model
        self.replay_buffer=replay_buffer
        self.policy=policy
        self.gamma=gamma
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.n
        self.tau=tau


    def train_model(self, training_data, training_label):
        self.trainable_model.fit(training_data, training_label, batch_size=64, verbose=0) #Training the network

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
        batch=self.replay_buffer.sample(64) #Sampling from replay buffer
        batchLen = len(batch)

        no_state = np.zeros(self.state_dim)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.compute_q_values(states)
        p_ = self.compute_q_values(states_)

        new_state = np.zeros((batchLen, self.state_dim))
        new_value = np.zeros((batchLen, self.action_dim))
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(p_[i])

            new_state[i] = s
            new_value[i] = t

        experiences=batch #Sampling from replay buffer
        states, actions, rewards, next_states=zip(*[[experience[0], experience[1], experience[2], experience[3]] for experience in experiences]) #Seperating states, actions, rewards and next states
        states=np.asarray(states) #Converting to numpy array
        place_holder_state=np.zeros(self.state_dim)
        next_states_=np.asarray([(place_holder_state if next_state is None else next_state) for next_state in next_states]) #Converting to numpy array
        q_values_for_states=self.compute_q_values(states)
        q_values_for_next_states=self.compute_q_values(next_states_, target=False) #Computing the max Q(S',A') for the using the target network
        for x in range(len(experiences)):
            y_true=rewards[x]
            if next_states[x] is not None:
                y_true +=self.gamma*(np.amax(q_values_for_next_states[x])) #Creating new target values for state action pair
            q_values_for_states[x][actions[x]]=y_true #Updating the old target values with the new values
        self.train_model(states, q_values_for_states)


    def compile(self, opt=Adam(lr=0.001)):
        pass
        # self.target_model=clone_model(self.trainable_model)
        # self.trainable_model.compile(loss='mse', optimizer=opt)
        # self.target_model.compile(loss='mse', optimizer=opt)


    def fit(self, number_of_epsiodes, batch_size):
        for episode in range(number_of_epsiodes): #Looping through total number of episodes
            total_reward=0 #The total reward an agent will get after an epsiode
            state=self.env.reset() #Resetting environment
            # if self.tau<=0 and episode%5000==0:
            #     print("Updating Target Model")
            #     self.update_target_model() #Update the target network after N episodes
            while True:
                #self.env.render()  #Rendering the environment
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
            print("Total reward ", total_reward)
