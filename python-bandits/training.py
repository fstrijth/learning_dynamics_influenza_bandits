from influenza_env import Influenza_env
from epsilon_greedy import Epsilon_Greedy_Agent
from UCB1 import UCB1_Agent
from softmax_boltzmann import Softmax_Boltzmann_Agent

from typing import Union, Iterable
from numpy import zeros
from random import randint

def train_agent(agent:Union[Epsilon_Greedy_Agent,UCB1_Agent,Softmax_Boltzmann_Agent],
             num_episodes:int,strategies:Iterable[str],env):
    eval_rewards = zeros(num_episodes)
    action_List = zeros(num_episodes)
    for t in range(num_episodes):
        #Train the agent
        action_train = agent.act(True)
        outcome_train = env.reward(strategies[action_train])
        rew_train = 1-outcome_train
        agent.learn(rew_train,action_train)
        #Calculate greedy action
        action_greedy = agent.act(False)
        outcome_greedy = env.reward(strategies[action_greedy])
        rew_greedy = 1-outcome_greedy
        eval_rewards[t] = rew_greedy
        action_List[t] = action_greedy
        if type(agent) == Epsilon_Greedy_Agent: #TODO: move this into epsilon-greedy class
            agent.epsilon = max(agent.epsilon_min,agent.epsilon_decay*agent.epsilon)
    return eval_rewards, action_List