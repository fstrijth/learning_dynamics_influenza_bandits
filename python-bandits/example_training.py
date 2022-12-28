from epsilon_greedy import Epsilon_Greedy_Agent
from UCB1 import UCB1_Agent
from plotting import strategies_violin_plots

from typing import Union
from numpy import zeros
from numpy.random import normal
from random import randint

class NormalGame:
    """
    Normal Game environment.
    """
    def __init__(self):
        self.num_agents = 1
        self.num_actions = 10
        self.avg = [randint(5,25) for _ in range(self.num_actions)]
        self.std = [randint(1,5) for _ in range(self.num_actions)]
    def act(self, action: int):
        """
        Method to perform an action in the Matrix Game and obtain the associated reward.
        :param action: The joint action.
        :return: The reward.
        """
        a = action
        rew = normal(self.avg[a],self.std[a])
        return rew


def train_agent(env,agent:Union[Epsilon_Greedy_Agent,UCB1_Agent],num_episodes:int):
    returns = zeros(num_episodes)
    for t in range(num_episodes):
        action = agent.act(True)
        rew = env.act(action)
        returns[t] = rew
        agent.learn(rew,action)
        if type(agent) == Epsilon_Greedy_Agent:
            agent.epsilon = max(agent.epsilon_min,agent.epsilon*agent.epsilon_decay)
    return returns

def outcome_distribution(env,action:int,num_test:int):
    dist = zeros(num_test)
    for i in range(num_test):
        dist[i] =  env.act(action)
    return dist

num_episodes = 1000
env = NormalGame()
agent = Epsilon_Greedy_Agent(env.num_actions,0.1) #test with epsilon-greedy
#agent = UCB1_Agent(env.num_actions,100) #or test with UCB1
temp = train_agent(env,agent,num_episodes)
opt_strategy = agent.act()
distribution = [outcome_distribution(env,action,1000) for action in range(env.num_actions)]
strategies = list(range(env.num_actions))
strategies_violin_plots(strategies,distribution,opt_strategy)
