import matplotlib.pyplot as plt
from typing import Iterable
from numpy import ndarray, mean, argmin, zeros, var
from math import ceil

def strategies_violin_plots(strategies:Iterable[int],distributions:Iterable[ndarray],title:str):
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
    plt.xlabel("Strategy",fontsize=14)
    plt.ylabel("Outcome",fontsize=14)
    plt.xticks(strategies,fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-1,len(strategies))
    plt.axhline(y=mean(distributions[opt_strategy]),c="r",linestyle="--")
    plt.title(title,fontsize=16)
    plt.show()

def reward_plot(rewards:Iterable[ndarray],agent_names:Iterable[str],title:str):
    """rewards = list of rewards over 1000 times steps averaged using 500 bandits for each agent type
    agent_names = names of the algorithms used for each agent"""
    plt.figure(figsize=(10,6))
    for i in range(len(rewards)):
        plt.plot(rewards[i])
    plt.xlabel("Iterations",fontsize=14)
    plt.ylabel("Average reward",fontsize=14)
    plt.legend(agent_names)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title,fontsize=16)
    plt.show()

def optimal_strat_plot(actions:Iterable[Iterable[ndarray]],agent_names:Iterable[str],optimal_strategy:int,title:str):
    """actions = list containing for each agent type a the actions done by the agent type
    over a pre-defined number agents, agent_names = names of the agent types"""
    optimal_perc = [zeros(actions[0][0].shape) for _ in agent_names]
    for ag in range(len(agent_names)):
        for action_list in actions[ag]:
            for t in range(len(action_list)):
                if action_list[t] == optimal_strategy:
                    optimal_perc[ag][t] += 1/len(actions[ag])
    plt.figure(figsize=(10,6))
    for ag in range(len(agent_names)):
        plt.plot(optimal_perc[ag])
    plt.xlabel("Iterations",fontsize=14)
    plt.ylabel("Optimal actions (%)",fontsize=14)
    plt.legend(agent_names)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title,fontsize=16)
    plt.show()

def variance_plot(strategies:Iterable[int],distributions:Iterable[ndarray],title:str):
    variance_List = []
    for strat in strategies:
        variance_List.append(var(distributions[strat]))
    plt.figure(figsize=(10,6))
    plt.bar(strategies,variance_List)
    plt.title(title,fontsize=16)
    plt.xlabel("Iterations",fontsize=14)
    plt.ylabel("Variance",fontsize=14)
    plt.xticks(strategies,fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-1,len(strategies))
    plt.show()
