from epsilon_greedy import Epsilon_Greedy_Agent
from UCB1 import UCB1_Agent

from typing import Union
from numpy import array, zeros, mean
from random import randint

class MatrixGame:
    """
    Matrix Game environment.
    """
    def __init__(self):
        self.num_agents = 1
        self.num_actions = 4
        self.stoch_matrix = array([(0,2),(-5,-75),(-2,-18),(4,-10)])
    def act(self, action: int):
        """
        Method to perform an action in the Matrix Game and obtain the associated reward.
        :param action: The joint action.
        :return: The reward.
        """
        a = action
        rew = self.stoch_matrix[a]
        #with probability 0.5, we choose the first reward and the second reward
        #with probability 0.5
        prob = randint(0,1)
        if prob < 0.5:
            return rew[0]
        else:
            return rew[1]


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

num_episodes = 1000
env = MatrixGame()
agent = Epsilon_Greedy_Agent(4,0.1) #test with epsilon-greedy
#agent = UCB1_Agent(4,100) #or test with UCB1
temp = train_agent(env,agent,num_episodes)
print(agent.act())