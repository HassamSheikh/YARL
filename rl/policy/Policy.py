import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon=epsilon

    def compute_action_probabilisties(self, q_values):
        number_of_actions=len(q_values)
        action_probabilites=np.ones(number_of_actions, dtype=float)*self.epsilon/number_of_actions
        best_action=np.argmax(q_values)
        action_probabilites[best_action]+=(1-self.epsilon)
        return action_probabilites

    def select_action(self, q_values):
        action_probabilites=self.compute_action_probabilisties(q_values)
        return np.random.choice(len(q_values), p=action_probabilites)

class GreedyPolicy:
    def select_action(self, q_values):
        return np.argmax(q_values)
