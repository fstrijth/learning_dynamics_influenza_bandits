import numpy as np
from random import randint

class Epsilon_Greedy_Agent():
    """Class that implements the epsilon-greedy bandit, with nbr_actions arms and constant epsilon"""
    def __init__(self,nbr_actions:int,epsilon_max:int,
                 epsilon_decay:float=0,epsilon_min:float=0):
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.nbr_actions = nbr_actions
        self.q_table = np.zeros(nbr_actions)
        self.repeats = np.zeros(nbr_actions) #number of times each action has been done
    def greedy_action(self):
        """Return the action associated to the highest q-value"""
        return np.argmax(self.q_table)
    def act(self,training:bool = False):
        """During training, explore with probability self.epsilon and exploit otherwise. During testing,
        use the greedy action."""
        if training:
            prob = randint(0,1)
            if prob < self.epsilon:
                return randint(0,self.nbr_actions-1)
            else:
               return self.greedy_action()
        else:
            return self.greedy_action()
    def learn(self,rew:float,action:int):
        """Adapt the estimated reward of a given action for a given reward"""
        k = self.repeats[action]
        q_old = self.q_table[action]
        q_new = (rew+q_old*k)/(k+1)
        self.q_table[action] = q_new
        self.repeats[action] += 1
