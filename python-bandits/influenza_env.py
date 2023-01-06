import numpy as np
from os.path import exists
import os
import random

class Influenza_env():
    def __init__(self, flute_dir="../FluTE-bandits", exec_path="scripts/bandits/pre-vaccination.sh", 
    work_dir="work_dir", rl_dir="rl_dir", bin="bin/flute", config_template="configs/bandits/pre-vaccination.config.mako"):
        self.flute_dir = flute_dir 
        self.exec_path = os.path.join(flute_dir, exec_path)
        self.work_dir = os.path.join(flute_dir, work_dir)
        self.rl_dir = os.path.join(flute_dir, rl_dir)
        self.bin = os.path.join(flute_dir, bin)
        self.config_template = config_template

        self.strategies = ["1,0,0,0,0","0,1,0,0,0","0,0,1,0,0","0,0,0,1,0","0,0,0,0,1","1,1,0,0,0","1,0,1,0,0",
              "1,0,0,1,0","1,0,0,0,1","0,1,1,0,0","0,1,0,1,0","0,1,0,0,1","0,0,1,1,0","0,0,1,0,1",
              "0,0,0,1,1","0,0,1,1,1","0,1,0,1,1","0,1,1,0,1","0,1,1,1,0","1,0,0,1,1","1,0,1,0,1",
              "1,0,1,1,0","1,1,0,0,1","1,1,0,1,0","1,1,1,0,0","0,1,1,1,1","1,0,1,1,1","1,1,0,1,1",
              "1,1,1,0,1","1,1,1,1,0","1,1,1,1,1","0,0,0,0,0"]
        self.outcome_distr = dict()
        for strat in self.strategies:
            self.outcome_distr[strat] = list()

    def run_simulation(self,r0:str,quarantine:str,vaccine_strategy:str,saving_folder:str,run_length:str="180",population:str="one",doses:str="100"):
        seed = str(random.randint(0,2000))
        os.system("./"+self.exec_path+" "+self.work_dir+" "+self.rl_dir+" "+self.bin+" "+self.config_template+
                " "+saving_folder+" "+seed+" "+r0+" "+run_length+" "+population+" "+doses+" "
                +quarantine+" "+vaccine_strategy)
        strat_vals = vaccine_strategy.split(",")
        strat_name = "".join(strat_vals)
        outcome_file = os.path.join(self.flute_dir, saving_folder, strat_name, "outcome.txt")
        with open(outcome_file,"r") as f:
            for line in f:
                last_line = line
        return float(last_line)

    def distribution(self,r0:str,quarantine:str,vaccine_strategy:str,saving_folder:str,
                    run_length:str="180",population:str="one",doses:str="100",num_samples:int=1000):
        strat_vals = vaccine_strategy.split(",")
        strat_name = "".join(strat_vals)
        total_samples = 0
        outcome_file = os.path.join(self.flute_dir, saving_folder, strat_name, "outcome.txt")
        if exists(outcome_file):
            with open(outcome_file, "r") as f:
                for line in f:
                    if total_samples == num_samples:
                        break
                    self.outcome_distr[vaccine_strategy].append(float(line))
                    total_samples += 1
        for i in range(total_samples,num_samples):
            outcome = self.run_simulation(r0,quarantine,vaccine_strategy,saving_folder,run_length,population,doses)
            self.outcome_distr[vaccine_strategy].append(outcome)
        return self.outcome_distr
    
    def reward(self, action):
        rewards = self.outcome_distr[action]
        return random.choice(rewards)



    
