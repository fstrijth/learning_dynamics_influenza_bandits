import os
import sys
import random

random.seed(10)

exec_path = "./../FluTE-bandits/scripts/bandits/pre-vaccination.sh"
work_dir = "../FluTE-bandits/work_dir"
flute_rl_dir = "../FluTE-bandits/rl_dir"
flute_bin_dir = "../FluTE-bandits/bin/flute"
config_template = "configs/bandits/pre-vaccination.config.mako"

def run_simulation(r0:str,quarantine:str,vaccine_strategy:str,saving_folder:str,run_length:str="180",population:str="one",doses:str="100"):
    seed = str(random.randint(0,sys.maxsize))
    os.system(exec_path+" "+work_dir+" "+flute_rl_dir+" "+flute_bin_dir+" "+config_template+
              " "+saving_folder+" "+seed+" "+r0+" "+run_length+" "+population+" "+doses+" "
              +quarantine+" "+vaccine_strategy)
    with open("../FluTE-bandits/"+saving_folder+"/outcome.txt","r") as f:
        for line in f:
            last_line = line
    return float(last_line)