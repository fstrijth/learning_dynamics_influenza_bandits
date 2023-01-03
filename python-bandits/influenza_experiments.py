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
envs = dict()
for r0 in ["1.3","1.4"]:
    for quarantine in ["0","1"]:
        saving_folder = "outcomes_r"+r0+"_quarantine"+quarantine

        #Run simulations in the FluTE influenza environment
        env = Influenza_env()
        envs[saving_folder] = env
        for strat in strategies:
            print(strat)
            env.distribution(r0,quarantine,strat,saving_folder,num_samples=num_samples)
        outcome_distributions = env.outcome_distr

        #Plot the results of the simulations
        #strategies_violin_plots(range(len(strategies)),list(outcome_distributions.values()),"Violin plot with r0="+r0+", quarantine="+quarantine)

num_bandits = 500
num_episodes = 1000
agent_names = ["epsilon-greedy","UCB1","Softmax-Boltzmann"]

def run_exp(agent_name, env):
    reward_t = np.zeros(num_episodes)
    action_t = list()
    for i in range(num_bandits):
        if i % 10 == 0:
            print("Agent nr: " + str(i))
        agent = None 
        if agent_name == "epsilon-greedy":
            agent = Epsilon_Greedy_Agent(len(strategies), 0.1)
        elif agent_name == "UCB1":
            agent = UCB1_Agent(len(strategies), c = 0.001)
        elif agent_name == "Softmax-Boltzmann":
            agent = Softmax_Boltzmann_Agent(len(strategies), 1.75)
        res = train_agent(agent, num_episodes, strategies, env)
        reward_t += res[0]/num_bandits 
        action_t.append(res[1])
    return (reward_t, action_t)

for r0 in ["1.3","1.4"]:
    for quarantine in ["0","1"]:
        saving_folder = "outcomes_r"+r0+"_quarantine"+quarantine
        env = envs[saving_folder]
        reward_list = [np.zeros(num_episodes) for _ in range(3)]
        action_list = [[] for _ in range(3)]
        print("r0: "+r0+", quarantine: "+quarantine)
        for i in range(3):
            print(agent_names[i])
            res = run_exp(agent_names[i], env)
            reward_list[i] += res[0]
            action_list[i].extend(res[1])
        reward_plot(reward_list,agent_names,"Average rewards with r0="+r0+", quarantine="+quarantine)
        optimal_strat_plot(action_list,agent_names,1, "Fraction of optimal action chosen with r0="+r0+", quarantine="+quarantine)

