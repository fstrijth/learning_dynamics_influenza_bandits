from influenza_env import Influenza_env
from plotting import strategies_violin_plots
import numpy as np

#Parameters
r0 = "1.3"
quarantine = "0"
strategies = ["1,0,0,0,0","0,1,0,0,0","0,0,1,0,0","0,0,0,1,0","0,0,0,0,1","1,1,0,0,0","1,0,1,0,0",
              "1,0,0,1,0","1,0,0,0,1","0,1,1,0,0","0,1,0,1,0","0,1,0,0,1","0,0,1,1,0","0,0,1,0,1",
              "0,0,0,1,1","0,0,1,1,1","0,1,0,1,1","0,1,1,0,1","0,1,1,1,0","1,0,0,1,1","1,0,1,0,1",
              "1,0,1,1,0","1,1,0,0,1","1,1,0,1,0","1,1,1,0,0","0,1,1,1,1","1,0,1,1,1","1,1,0,1,1",
              "1,1,1,0,1","1,1,1,1,0","1,1,1,1,1","0,0,0,0,0"]
saving_folder = "outcomes_r"+r0+"_quarantine"+quarantine
num_samples = 1000

#Run simulations in the FluTE influenza environment
env = Influenza_env()
for strat in strategies:
    print(strat)
    env.distribution(r0,quarantine,strat,saving_folder,num_samples=num_samples)
outcome_distributions = env.outcome_distr

#Plot the results of the simulations
strategies_violin_plots(range(len(strategies)),list(outcome_distributions.values()))


#TODO: train epsilon greedy with epsilon value from paper

#TODO: train UCB with c = 2

#TODO: finetune boltzmann with a big enough range of temperatures
