import numpy as np
import sys
from math import log, sqrt
import random

def upper_confidence_bound(avg:float,n_i:int,n:int,c:float):
    """Compute the upper confidence bound for a given action of estimated reward avg,
    having been played n_i times out of n, where c is the UCB1 agent's parameter"""
    if n_i == 0:
        #if the action hasn't been played yet (or if no actions at all have been played
        #yet) assign value sys.maxsize
        return sys.maxsize
    else:
        return avg+sqrt(c*log(n)/n_i)

class UCB1_Agent():
    """Class that implements the UCB1 bandit with parameter c"""
    def __init__(self,nbr_actions:int,c:float=2,step_size=0.1):
        self.c = c
        self.total = 0 #total number of actions done so far
        self.nbr_actions = nbr_actions
        self.step_size=0.1
        self.repeats = np.zeros(nbr_actions) #number of times each action has been done
        self.q_table = np.zeros(nbr_actions)+1
    def greedy_action(self):
        """Return the action associated to the highest q-value"""
        return np.argmax(self.q_table)
    def act(self,training:bool=False):
        """During training, choose the action with the highest upper confidence bound. During testing,
        use the greedy action."""
        if training:
            upper_conf_bounds = np.zeros(self.nbr_actions)
            for action in range(self.nbr_actions):
                upper_confidence_bound(self.q_table[action], self.repeats[action], self.total, self.c)
            highest_bounds = [0]
            for i in range(len(upper_conf_bounds)):
                if upper_conf_bounds[i] > upper_conf_bounds[highest_bounds[0]]:
                    highest_bounds = [upper_conf_bounds[i]]
                elif upper_conf_bounds[i] == upper_conf_bounds[highest_bounds[0]]:
                    highest_bounds.append(i)
            #upper_conf_bounds = np.array(
            #    [upper_confidence_bound(self.q_table[action],self.repeats[action],self.total,self.c)
            #     for action in range(self.nbr_actions)])
            return random.choice(highest_bounds) #using a simple argmax of the upper_conf_bounds might introduce bias when 
        else:
            return self.greedy_action()
    def learn(self,rew:float,action:int):
        """Adapt the estimated reward of a given action for a given reward"""
        q_old = self.q_table[action]
        q_new = q_old+self.step_size*(rew-q_old)
        self.q_table[action] = q_new
        self.repeats[action] += 1
        self.total += 1
