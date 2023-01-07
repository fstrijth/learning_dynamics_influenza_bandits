from plotting import reward_plot, optimal_strat_plot
from softmax_boltzmann import Softmax_Boltzmann_Agent
from training import train_agent
from influenza_env import Influenza_env

import numpy as np

#Parameters

strategies = ["1,0,0,0,0","0,1,0,0,0","0,0,1,0,0","0,0,0,1,0","0,0,0,0,1","1,1,0,0,0","1,0,1,0,0",
              "1,0,0,1,0","1,0,0,0,1","0,1,1,0,0","0,1,0,1,0","0,1,0,0,1","0,0,1,1,0","0,0,1,0,1",
              "0,0,0,1,1","0,0,1,1,1","0,1,0,1,1","0,1,1,0,1","0,1,1,1,0","1,0,0,1,1","1,0,1,0,1",
              "1,0,1,1,0","1,1,0,0,1","1,1,0,1,0","1,1,1,0,0","0,1,1,1,1","1,0,1,1,1","1,1,0,1,1",
              "1,1,1,0,1","1,1,1,1,0","1,1,1,1,1","0,0,0,0,0"]
temperatures = [0.25,0.75,1.25,1.75]
num_bandits = 750
num_episodes = 1000
reward_List = [np.zeros(num_episodes) for t in temperatures]
action_List = [[] for t in temperatures]
for r0 in ["1.3","1.4"]:
    for quarantine in ["0","1"]:
        saving_folder = "outcomes_r"+r0+"_quarantine"+quarantine
        env = Influenza_env()
        print("r_0 = "+ r0+", quarantine = "+quarantine)
        for strat in strategies:
            env.distribution(r0,quarantine,strat,saving_folder,num_samples=2000)
        for j in range(len(temperatures)):
            t = temperatures[j]
            print("temperature = ",t)
            reward_t = np.zeros(num_episodes)
            action_t = []
            for i in range(num_bandits):
                agent = Softmax_Boltzmann_Agent(len(strategies),t)
                res = train_agent(agent,num_episodes,strategies,env)
                action_t.append(res[1])
                reward_t += res[0]/num_bandits
            reward_List[j] += reward_t / 4
            action_List[j].extend(action_t)
temp_names = ["t="+str(t) for t in temperatures]
reward_plot(list(reward_List),temp_names,"Softmax")
optimal_strat_plot(list(action_List),temp_names,1,"Softmax")
