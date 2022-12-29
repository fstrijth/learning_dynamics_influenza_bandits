import matplotlib.pyplot as plt
from typing import Iterable
from numpy import ndarray, mean

def strategies_violin_plots(strategies:Iterable[int],distributions:Iterable[ndarray],
                            opt_strategy:int):
    """strategies is a list of integers representing the predictive strategies,
    distributions is a list containing the distribution of the outcome (not the reward) of each
    of the strategies in strategies. opt_strategy is the integer representing the optimal strategy"""
    means = [mean(distributions[strat]) for strat in strategies]
    plt.violinplot(distributions,strategies)
    plt.scatter(strategies,means,c="r",marker="D")
    plt.xlabel("Strategy",fontsize=12)
    plt.ylabel("Outcome",fontsize=12)
    plt.xticks(strategies,fontsize=12)
    plt.yticks(fontsize=12)
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
