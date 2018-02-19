import random
import warnings
import numpy as np
from rl.util import *
from keras.optimizers import Adam, RMSprop
from keras import losses
import pdb

class DDPGAgent:
    def __init__(self, env, actor, critic, replay_buffer, random_process, gamma=0.99, batch_size=64, tau=0, render=False):
        if len(critic.outputs) > 1:
            raise ValueError('Critic has more than 1 outputs. DDPG critic should have 1 output to predict the state-action value')
        if len(critic.inputs) != 2:
            raise ValueError('Critic expects only state and action as input')
        if len(actor.outputs) > 1:
            raise ValueError('Actor has more than 1 outputs. DDPG actor should have 1 output to predict the action')
        if len(actor.inputs) != 1:
            raise ValueError('Actor expects only state as input')
        self.env=env
        self.actor_trainable_model=actor
        self.critic_trainable_model=critic
        self.replay_buffer=replay_buffer
        self.random_process=random_process
        self.gamma=gamma
        self.batch_size=batch_size
        self.tau=tau
        self.render=render
        self.state_dim, self.action_dim = find_state_and_action_dimension(env)
        if self.render:
            warnings.warn("Warning: Rendering environment will make the training extremely slow")
        if self.tau>0:
            warnings.warn("Warning: Polyvak Averaging will be used to update target model at every step")

    def compile(self, opt=[RMSprop(lr=0.00025), RMSprop(lr=0.00025)] , loss=['mse','mse']):
        """Set optimizers and loss functions for Actor and Critic Model """
        if not hasattr(self.actor_trainable_model, 'loss') or not hasattr(self.actor_trainable_model, 'optimizer') or not hasattr(self.critic_trainable_model, 'loss') or not hasattr(self.critic_trainable_model, 'optimizer'):
            warnings.warn("Warning: At least one of your models is missing either an optimizer or loss function")
            if len(opt) != 2 or len(loss) != 2:
                raise ValueError('Provide 2 optimizers and 2 loss functions')
        self.actor_target_model=clone_model(self.actor_trainable_model)
        self.critic_target_model=clone_model(self.critic_trainable_model)

        self.actor_trainable_model.compile(loss=loss[0], optimizer=opt[0])
        self.critic_trainable_model.compile(loss=loss[1], optimizer=opt[1])

        self.actor_target_model.compile(loss=loss[0], optimizer=opt[0])
        self.critic_target_model.compile(loss=loss[1], optimizer=opt[1])

    def train_model(self, training_data, training_label, batch_size=64):
        self.trainable_model.fit(training_data, training_label, batch_size=self.batch_size, verbose=0) #Training the network

    def update_target_model(self):
        if self.tau>0:
            self.target_model=update_model_by_polyak_average(self.target_model, self.trainable_model, self.tau) #Updating the target network using the Polyak Averaging
        else:
            self.target_model=clone_weights(self.target_model, self.trainable_model) #Updating the target network in case Polyak Averaging is not used

    def select_action(self, state, target=False):
        return self.actor_trainable_model.predict(state.reshape(1, self.state_dim)).flatten() + self.random_process()

    def select_action_from_target_actor(self, state):
        return self.actor_target_model.predict(state.reshape(1, self.state_dim)).flatten()

    def compute_q_values(self, state, target=False):
        if target:
            action = self.select_action_from_target_actor(state)
            return self.critic_target_model([state, action])


    def experience_replay(self):
        experiences = self.replay_buffer.sample(self.batch_size)
        import pdb; pdb.set_trace()
        batch_len = self.batch_size
        training_data=np.zeros((batch_len, self.state_dim))
        training_label=np.zeros((batch_len, 1))
        place_holder_state=np.zeros(self.state_dim)
        for index, (state, action, reward, next_state) in enumerate(experiences):
            if next_state is None:
                y_target = reward
            else:
                y_target = reward + (self.gamma * self.compute_q_values(next_state, True))

        # ff_states = experiences[:,[0]]
        # states=[experience for experience in experiences] #Seperating states, actions, rewards and next states
        # states=np.asarray(states) #Converting to numpy array
        # place_holder_state=np.zeros(self.state_dim)
        # next_states_=np.asarray([(place_holder_state if next_state is None else next_state) for next_state in next_states]) #Converting to numpy array
        #
        # q_values_for_states=self.compute_q_values(states)
        #
        # q_values_for_next_states=self.compute_q_values(next_states_, True)
        #
        # batch_len=len(experiences)
        # training_data=np.zeros((batch_len, self.state_dim))
        # training_label=np.zeros((batch_len, self.action_dim))
        # index=0
        # for state, action, reward, next_state in experiences:
        #     y_true = q_values_for_states[index]
        #     pdb.set_trace()
        #     if next_state is None:
        #         y_true[action] = reward
        #     else:
        #         y_true[action] = reward + (self.gamma * np.amax(q_values_for_next_states[index]))
        #     training_data[index]=state
        #     training_label[index]=y_true
        #     index+=1
        # self.train_model(training_data, training_label)


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
                self.replay_buffer.add_to_buffer(np.asarray((state, action, reward, next_state))) #Adding experience to replay buffer
                state=next_state
                total_reward +=reward
                if episode > 5:
                    self.experience_replay() #Experience replay step
                if done:
                    break
            # if self.tau>0 or episode%self.target_model_update_interval==0:
            #     self.update_target_model()
            print("Total reward ", total_reward)
