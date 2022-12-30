from epsilon_greedy import Epsilon_Greedy_Agent
from UCB1 import UCB1_Agent
from softmax_boltzmann import Softmax_Boltzmann_Agent
from plotting import strategies_violin_plots, reward_plot

from typing import Union
from numpy import zeros
from numpy.random import normal
from random import uniform

#This code tests our agents and plotting function in a simpler normal game environment

class NormalGame:
    """
    Normal Game environment.
    """
    def __init__(self):
        self.num_agents = 1
        self.num_actions = 10
        self.avg = [uniform(5,10) for _ in range(self.num_actions-1)]+[20]
        self.std = [uniform(1,2) for _ in range(self.num_actions)]
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
    eval_returns = zeros(num_episodes)
    for t in range(num_episodes):
        action = agent.act(True)
        rew = env.act(action)
        agent.learn(rew,action)
        action = agent.act()
        rew = env.act(action)
        eval_returns[t] = rew
        if type(agent) == Epsilon_Greedy_Agent:
            agent.epsilon = max(agent.epsilon_min,agent.epsilon*agent.epsilon_decay)
    return eval_returns

def outcome_distribution(env,action:int,num_test:int):
    dist = zeros(num_test)
    for i in range(num_test):
        dist[i] =  env.act(action)
    return dist

num_episodes = 1000
env = NormalGame()
#agent = Epsilon_Greedy_Agent(env.num_actions,0.1) #test with epsilon-greedy
#agent = UCB1_Agent(env.num_actions) #or test with UCB1
agent = Softmax_Boltzmann_Agent(env.num_actions,10)

#Examples violin plots
train_agent(env,agent,num_episodes)
opt_strategy = agent.act()
distribution = [outcome_distribution(env,action,1000) for action in range(env.num_actions)]
strategies = list(range(env.num_actions))
strategies_violin_plots(strategies,distribution,opt_strategy)

#Examples average reward of agents
num_bandits = 500
num_episodes = 1000
rewards_greedy = zeros(num_episodes)
for _ in range(num_bandits):
    agent = Epsilon_Greedy_Agent(env.num_actions,0.1)
    rewards_greedy += train_agent(env,agent,num_episodes)/num_bandits
rewards_UCB1 = zeros(num_episodes)
for _ in range(num_bandits):
    agent = UCB1_Agent(env.num_actions)
    temp_rewards = train_agent(env,agent,num_episodes)
    rewards_UCB1 += train_agent(env,agent,num_episodes)/num_bandits
rewards_boltzmann = zeros(num_episodes)
for _ in range(num_bandits):
    agent = Softmax_Boltzmann_Agent(env.num_actions,2)
    temp_rewards = train_agent(env,agent,num_episodes)
    rewards_boltzmann += train_agent(env,agent,num_episodes)/num_bandits

reward_plot([rewards_greedy,rewards_UCB1,rewards_boltzmann],["epsilon-greedy","UCB1","Softmax-Boltzmann"])

