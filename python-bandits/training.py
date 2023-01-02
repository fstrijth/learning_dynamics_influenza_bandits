from influenza_env import Influenza_env
from epsilon_greedy import Epsilon_Greedy_Agent
from UCB1 import UCB1_Agent
from softmax_boltzmann import Softmax_Boltzmann_Agent

from typing import Union, Iterable
from numpy import zeros
from random import randint

def train_agent(agent:Union[Epsilon_Greedy_Agent,UCB1_Agent,Softmax_Boltzmann_Agent],
             num_episodes:int,strategies:Iterable[str],saving_dir:str):
    strat_Vals = [strat.split(",") for strat in strategies]
    strat_Names = ["".join(strat) for strat in strat_Vals]
    eval_rewards = zeros(num_episodes)
    action_List = zeros(num_episodes)
    outcome_List = {i:[] for i in strat_Names}
    for strat in strat_Names:
        filename = "../FluTE-bandits/"+saving_dir+"/"+strat+"/outcome.txt"
        with open(filename,"r") as f:
            for line in f:
                outcome_List[strat].append(float(line))
    for t in range(num_episodes):
        action = agent.act(True)
        outcome = outcome_List[strat_Names[action]][randint(0,len(outcome_List[strat_Names[action]])-1)]
        rew = 1-outcome
        agent.learn(rew,action)
        action = agent.act()
        outcome = outcome_List[strat_Names[action]][randint(0, len(outcome_List[strat_Names[action]]) - 1)]
        rew = 1-outcome
        eval_rewards[t] = rew
        action_List[t] = action
        if type(agent) == Epsilon_Greedy_Agent:
            agent.epsilon = max(agent.epsilon_min,agent.epsilon_decay*agent.epsilon)
    return eval_rewards, action_List