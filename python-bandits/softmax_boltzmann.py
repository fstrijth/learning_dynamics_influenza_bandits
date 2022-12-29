import numpy as np
from random import uniform
from math import exp

class Softmax_Boltzmann_Agent():
    """Class that implements the softmax bandit using a Boltzmann function, with nbr_actions arms"""
    def __init__(self,nbr_actions:int,temperature:float,):
        self.nbr_actions = nbr_actions
        self.temperature = temperature
        self.q_table = np.zeros(nbr_actions)
        self.repeats = np.zeros(nbr_actions) #number of times each action has been done
    def greedy_action(self):
        """Return the action associated to the highest q-value"""
        return np.argmax(self.q_table)
    def act(self,training:bool = False):
        """During training, select each action with a probability defined by the Boltzmann
        function. During testing, use the greedy action."""
        if training:
            exp_q_sum = sum([exp(self.q_table[i]/self.temperature) for i in range(self.nbr_actions)])
            prob_boltzmann = [exp(self.q_table[i]/self.temperature)/exp_q_sum for i in range(self.nbr_actions)]
            prob = uniform(0,1)
            current_prob = 0
            action = None
            for i in range(self.nbr_actions):
                next_prob = current_prob + prob_boltzmann[i]
                if prob >= current_prob and prob < next_prob:
                    action = i
                    break
                current_prob = next_prob
            if action == None:
                action = self.nbr_actions-1
            return action
        else:
            return self.greedy_action()
    def learn(self,rew:float,action:int):
        """Adapt the estimated reward of a given action for a given reward"""
        k = self.repeats[action]
        q_old = self.q_table[action]
        q_new = (rew+q_old*k)/(k+1)
        self.q_table[action] = q_new
        self.repeats[action] += 1