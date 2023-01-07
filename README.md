# Efficient evaluation of influenza mitigation strategies using preventative bandits


## The FluTE environment
We used this FluTE implementation https://github.com/vub-ai-lab/FluTE-bandits.

How to install it:
Under bin/flute, there is already a linux executable. If it is not compatible with your machine, you will have to compile the c++ code with cmake.
Before running the code you will have to install the python package mako:
pip install mako

FluTE can be started on its own with the pre-vaccination.sh script:
./scripts/bandits/pre-vaccination.sh work_dir rl_dir bin/flute configs/bandits/pre-vaccination.config.mako 10 1.3 10 la 100 1,0,0,0,0


## Python bandits (our code)
Our experiments can be started by running the file influenza_exp.py. It uses previously generated simulation data. 
