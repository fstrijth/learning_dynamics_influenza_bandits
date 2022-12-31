import matplotlib.pyplot as plt
from typing import Iterable
from numpy import ndarray, mean, argmin, zeros

strategies = ["1,0,0,0,0","0,1,0,0,0","0,0,1,0,0","0,0,0,1,0","0,0,0,0,1","1,1,0,0,0","1,0,1,0,0",
              "1,0,0,1,0","1,0,0,0,1","0,1,1,0,0","0,1,0,1,0","0,1,0,0,1","0,0,1,1,0","0,0,1,0,1",
              "0,0,0,1,1","0,0,1,1,1","0,1,0,1,1","0,1,1,0,1","0,1,1,1,0","1,0,0,1,1","1,0,1,0,1",
              "1,0,1,1,0","1,1,0,0,1","1,1,0,1,0","1,1,1,0,0","0,1,1,1,1","1,0,1,1,1","1,1,0,1,1",
              "1,1,1,0,1","1,1,1,1,0","1,1,1,1,1","0,0,0,0,0"]

def strategies_violin_plots(strategies:Iterable[int],distributions:Iterable[ndarray]):
    """strategies is a list of integers representing the predictive strategies,
    distributions is a list containing the distribution of the outcome (not the reward) of each
    of the strategies in strategies. opt_strategy is the integer representing the optimal strategy"""
    means = [mean(distributions[strat]) for strat in strategies]
    opt_strategy = argmin(means)
    plt.figure(figsize=(10,6))
    parts = plt.violinplot(distributions,strategies,widths=0.8,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('teal')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    plt.scatter(strategies,means,c="r",marker="D")
    plt.xlabel("Strategy",fontsize=12)
    plt.ylabel("Outcome",fontsize=12)
    plt.xticks(strategies,fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-1,32)
    plt.axhline(y=mean(distributions[opt_strategy]),c="r",linestyle="--")
    plt.show()

def reward_plot(rewards:Iterable[ndarray],agent_names:Iterable[str]):
    """rewards = list of rewards over 1000 times steps averaged using 500 bandits for each agent type
    agent_names = names of the algorithms used for each agent"""
    for i in range(len(rewards)):
        plt.plot(rewards[i])
    plt.xlabel("Iterations",fontsize=12)
    plt.ylabel("Average reward",fontsize=12)
    plt.legend(agent_names)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def optimal_strat_plot(actions:Iterable[Iterable[ndarray]],agent_names:Iterable[str],optimal_strategy:int):
    """actions = list containing for each agent type a the actions done by the agent type
    over a pre-defined number agents, agent_names = names of the agent types"""
    optimal_perc = [zeros(actions[0][0].shape) for _ in agent_names]
    for ag in range(len(agent_names)):
        for action_list in actions[ag]:
            for t in range(len(action_list)):
                if action_list[t] == optimal_strategy:
                    optimal_perc[ag][t] += 1/len(actions[ag])
    for ag in range(len(agent_names)):
        plt.plot(optimal_perc[ag])
    plt.xlabel("Iterations",fontsize=12)
    plt.ylabel("Optimal actions (%)",fontsize=12)
    plt.legend(agent_names)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()