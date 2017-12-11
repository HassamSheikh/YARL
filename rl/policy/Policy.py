import numpy as np
import random
import math

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon=epsilon

    def compute_action_probabilities(self, q_values):
        number_of_actions=len(q_values)
        action_probabilites=np.ones(number_of_actions, dtype=float)*self.epsilon/number_of_actions
        best_action=np.argmax(q_values)
        action_probabilites[best_action]+=(1-self.epsilon)
        return action_probabilites

    def select_action(self, q_values):
        number_of_actions=len(q_values)
        action_probabilites=self.compute_action_probabilities(q_values)
        return np.random.choice(len(q_values), p=action_probabilites)

class GreedyPolicy:
    def select_action(self, q_values):
        return np.argmax(q_values)

class DecayingEpsilonGreedyPolicy:
    def __init__(self, max_epsilon=1.0, min_epsilon=0.01, decaying_rate=0.001):
        self.epsilon=max_epsilon
        self.max_epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.decaying_rate=decaying_rate
        self.steps=0

    def compute_action_probabilities(self, q_values):
        number_of_actions=len(q_values)
        action_probabilites=np.ones(number_of_actions, dtype=float)*self.epsilon/number_of_actions
        best_action=np.argmax(q_values)
        action_probabilites[best_action]+=(1-self.epsilon)
        return action_probabilites

    def select_action(self, q_values):
        self.epsilon = self.min_epsilon+(self.max_epsilon-self.min_epsilon)*math.exp(-self.decaying_rate*self.steps)
        self.steps += 1
        number_of_actions=len(q_values)
        action_probabilites=self.compute_action_probabilities(q_values)
        return np.random.choice(len(q_values), p=action_probabilites)
