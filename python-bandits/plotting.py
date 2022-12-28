import matplotlib.pyplot as plt
from typing import Iterable
from numpy import ndarray, mean

def strategies_violin_plots(strategies:Iterable[int],distributions:Iterable[ndarray],
                            opt_strategy:int):
    """strategies is a list of integers representing the predictive strategies,
    distributions is a list containing the distribution of the outcome of each
    of the strategies in strategies. opt_strategy is the integer representing the optimal strategy"""
    means = [mean(distributions[strat]) for strat in strategies]
    plt.violinplot(distributions,strategies)
    plt.scatter(strategies,means,c="r",marker="D")
    plt.xlabel("Strategy")
    plt.ylabel("Outcome")
    plt.xticks(strategies)
    plt.axhline(y=mean(distributions[opt_strategy]),c="r",linestyle="--")
    plt.show()