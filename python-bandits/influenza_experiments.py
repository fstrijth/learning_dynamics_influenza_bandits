from influenza_env import Influenza_env
from plotting import strategies_violin_plots, reward_plot, optimal_strat_plot
from epsilon_greedy import Epsilon_Greedy_Agent
from UCB1 import UCB1_Agent
from softmax_boltzmann import Softmax_Boltzmann_Agent
from training import train_agent

import numpy as np


strategies = ["1,0,0,0,0","0,1,0,0,0","0,0,1,0,0","0,0,0,1,0","0,0,0,0,1","1,1,0,0,0","1,0,1,0,0",
              "1,0,0,1,0","1,0,0,0,1","0,1,1,0,0","0,1,0,1,0","0,1,0,0,1","0,0,1,1,0","0,0,1,0,1",
              "0,0,0,1,1","0,0,1,1,1","0,1,0,1,1","0,1,1,0,1","0,1,1,1,0","1,0,0,1,1","1,0,1,0,1",
              "1,0,1,1,0","1,1,0,0,1","1,1,0,1,0","1,1,1,0,0","0,1,1,1,1","1,0,1,1,1","1,1,0,1,1",
              "1,1,1,0,1","1,1,1,1,0","1,1,1,1,1","0,0,0,0,0"]
num_samples = 1500

for r0 in ["1.3","1.4"]:
    for quarantine in ["0","1"]:
        saving_folder = "outcomes_r"+r0+"_quarantine"+quarantine

        #Run simulations in the FluTE influenza environment
        env = Influenza_env()
        for strat in strategies:
            print(strat)
            env.distribution(r0,quarantine,strat,saving_folder,num_samples=num_samples)
        outcome_distributions = env.outcome_distr

        #Plot the results of the simulations
        strategies_violin_plots(range(len(strategies)),list(outcome_distributions.values()))

num_bandits = 500
num_episodes = 1000
agent_names = ["epsilon-greedy","UCB1","Softmax-Boltzmann"]
for r0 in ["1.3","1.4"]:
    for quarantine in ["0","1"]:
        saving_folder = "outcomes_r"+r0+"_quarantine"+quarantine
        reward_List = [np.zeros(num_episodes) for _ in range(3)]
        action_List = [[] for _ in range(3)]
        print(r0,quarantine)
        reward_t = np.zeros(num_episodes)
        action_t = []
        for i in range(num_bandits):
            print(i)
            agent = Epsilon_Greedy_Agent(len(strategies),0.1)
            res = train_agent(agent,num_episodes,strategies,saving_folder)
            action_t.append(res[1])
            reward_t += res[0]/num_bandits
        reward_List[0] += reward_t
        action_List[0].extend(action_t)
        reward_t = np.zeros(num_episodes)
        action_t = []
        for i in range(num_bandits):
            print(i)
            agent = UCB1_Agent(len(strategies))
            res = train_agent(agent,num_episodes,strategies,saving_folder)
            action_t.append(res[1])
            reward_t += res[0]/num_bandits
        reward_List[1] += reward_t
        action_List[1].extend(action_t)
        reward_t = np.zeros(num_episodes)
        action_t = []
        for i in range(num_bandits):
            print(i)
            agent = Softmax_Boltzmann_Agent(len(strategies),1.75)
            res = train_agent(agent,num_episodes,strategies,saving_folder)
            action_t.append(res[1])
            reward_t += res[0]/num_bandits
        reward_List[2] += reward_t
        action_List[2].extend(action_t)
        reward_plot(reward_List,agent_names)
        optimal_strat_plot(action_List,agent_names,1)


