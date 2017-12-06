import random
import numpy as np
from rl.util import generator

class DQNAgent:
    def __init__(self, env, model, gamma, replay_buffer, policy):
        self.env=env
        self.model=model
        self.replay_buffer=replay_buffer
        self.policy=policy
        self.gamma=gamma
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.n

    def train_model(self, training_data, training_label):
        self.model.train_on_batch(training_data, training_label)

    def compute_q_values(self, states):
        if (states.shape)== (self.state_dim,):
            return self.model.predict(states.reshape(1, self.state_dim)).flatten()
        return self.model.predict(states)

    def experience_replay(self, experiences):
        states, actions, rewards, next_states=zip(*[[experience[0], experience[1], experience[2], experience[3]] for experience in experiences])
        states=np.asarray(states)
        place_holder_state=np.zeros(self.state_dim)
        next_states_=np.asarray([(place_holder_state if next_state is None else next_state) for next_state in next_states])
        q_values_for_states=self.compute_q_values(states)
        q_values_for_next_states=self.compute_q_values(next_states_)
        for x in generator(len(experiences)):
            y_true=rewards[x]
            if next_states[x] is not None:
                y_true +=self.gamma*(np.amax(q_values_for_next_states[x]))
            q_values_for_states[x][actions[x]]=y_true
        self.train_model(states, q_values_for_states)

    def fit(self, number_of_epsiodes, batch_size):
        for _ in generator(number_of_epsiodes):
            total_reward=0
            state=self.env.reset()
            while True:
                #self.env.render()
                q_values_for_state=self.compute_q_values(state)
                action=self.policy.select_action(q_values_for_state)
                next_state, reward, done, _=self.env.step(action)
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
