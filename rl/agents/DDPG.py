import random
import warnings
import numpy as np
from rl.util import *
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras import backend as K
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
        # self.actor_target_model=clone_model(self.actor_trainable_model)
        # self.critic_target_model=clone_model(self.critic_trainable_model)

        self.action_gradients = K.tf.gradients(self.critic_trainable_model.output, self.critic_trainable_model.input_layers[1].input, name="critic_gradients_wrt_action")

        self.action_gradient_place_holder = K.tf.placeholder(K.tf.float32, [None, self.action_dim])
        self.unnormalized_actor_gradients = K.tf.gradients(self.actor_trainable_model.output, self.actor_trainable_model.trainable_weights, -self.action_gradient_place_holder, name="actor_gradients_wrt_network_params")

        self.actor_gradients = list(map(lambda x: K.tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = K.tf.train.AdamOptimizer(0.1).apply_gradients(zip(self.actor_gradients, self.actor_trainable_model.trainable_weights))


        self.actor_trainable_model.compile(loss=loss[0], optimizer=opt[0])
        self.critic_trainable_model.compile(loss=loss[1], optimizer=opt[1])
        self.sess = K.get_session()
        # self.actor_target_model.compile(loss=loss[0], optimizer=opt[0])
        # self.critic_target_model.compile(loss=loss[1], optimizer=opt[1])

    def train_critic_model(self, training_data, training_label, batch_size=64):
        self.critic_trainable_model.fit(training_data, training_label, batch_size=self.batch_size, verbose=0) #Training the network

    # def train_actor_model(self, ):
    #      self.sess.run(self.optimize, feed_dict={
    #         self.inputs: inputs,
    #         self.action_gradient_place_holder: a_gradient
    #     })


    def update_target_model(self):
        if self.tau>0:
            self.target_model=update_model_by_polyak_average(self.target_model, self.trainable_model, self.tau) #Updating the target network using the Polyak Averaging
        else:
            self.target_model=clone_weights(self.target_model, self.trainable_model) #Updating the target network in case Polyak Averaging is not used

    def select_action(self, state):
        return self.actor_trainable_model.predict(state.reshape(1, self.state_dim)).flatten() + self.random_process()

    def compute_actor_gradients(self, states, critic_action_gradients):
         return self.sess.run(self.actor_gradients, feed_dict={
            self.actor_trainable_model.input_layers[0].input: states,
            self.action_gradient_place_holder: critic_action_gradients
        })[0]

    def compute_critic_gradients(self, states, actions):
         return self.sess.run(self.action_gradients, feed_dict={
            self.critic_trainable_model.input_layers[0].input: states,
            self.critic_trainable_model.input_layers[1].input: actions
        })[0]


    def compute_q_values(self, states, target=False):
        if target:
            actions = self.actor_target_model.predict(states)
            return self.critic_target_model.predict([states, actions]) #Querying target network for Q values of multiple states
        actions = self.actor_trainable_model.predict(states)
        return self.critic_trainable_model.predict([states, actions])

    def experience_replay(self):
        experiences=self.replay_buffer.sample(self.batch_size)
        states=np.asarray([e[0] for e in experiences])
        actions=np.asarray([e[1] for e in experiences])
        rewards=np.asarray([e[2] for e in experiences])
        next_states=([e[3] for e in experiences])#Seperating states, actions, rewards and next states
        place_holder_state=np.zeros(self.state_dim)
        next_states_=np.asarray([(place_holder_state if next_state is None else next_state) for next_state in next_states])
        q_values_for_next_states=self.compute_q_values(next_states_, True)
        training_label=np.zeros((self.batch_size, 1))
        for index, next_state in enumerate(next_states):
            if next_state is None:
                y_target = rewards[index]
            else:
                y_target = rewards[index] + (self.gamma * q_values_for_next_states[index])
            training_label[index]=y_target
        self.train_critic_model([states, actions], training_label)
        test = self.compute_critic_gradients(states, actions)
        ff = self.compute_actor_gradients(states, -test)
        import pdb; pdb.set_trace()

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
                state=next_state
                total_reward +=reward
                if episode > 50:
                    self.experience_replay() #Experience replay step
                if done:
                    break
            # if self.tau>0 or episode%self.target_model_update_interval==0:
            #     self.update_target_model()
            print("Total reward ", total_reward)
