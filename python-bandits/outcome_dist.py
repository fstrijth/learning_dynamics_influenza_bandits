from pipeline import run_simulation
from plotting import strategies_violin_plots

import numpy as np
from os.path import exists

def distribution(r0:str,quarantine:str,vaccine_strategy:str,saving_folder:str,
                 run_length:str="180",population:str="one",doses:str="100",num_samples:int=1000):
    strat_vals = vaccine_strategy.split(",")
    strat_name = "".join(strat_vals)
    dist_outcomes = np.zeros(num_samples)
    total_samples = 0
    if exists("../FluTE-bandits/" + saving_folder + "/" + strat_name + "/outcome.txt"):
        with open("../FluTE-bandits/" + saving_folder + "/" + strat_name + "/outcome.txt", "r") as f:
            for line in f:
                if total_samples == num_samples:
                    break
                dist_outcomes[total_samples] = float(line)
                total_samples += 1
    for i in range(total_samples,num_samples):
        print(i)
        dist_outcomes[i] = run_simulation(r0,quarantine,vaccine_strategy,saving_folder,run_length,population,doses)
    return dist_outcomes
