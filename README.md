# learning_dynamics_influenza_bandits


## The FluTE environment
Used this FluTE implementation https://github.com/vub-ai-lab/FluTE-bandits. Apart from changing the starting scripts around a bit, the code remains the same

How to install it:
Under bin/flute, there is already a linux executable. If it is not compatible with your machine, you will have to compile the c++ code with cmake.
Before running the code you will have to install the python package mako (pip install mako)

Example of how to start FluTE:
./scripts/bandits/pre-vaccination.sh work_dir rl_dir bin/flute configs/bandits/pre-vaccination.config.mako 10 1.3 10 la 100 1,2,3,4,5


## Scala bandits (original code)
How to install:
sudo apt-get scala
Then follow the instructions here to install sbt https://www.scala-sbt.org/1.x/docs/Installing-sbt-on-Linux.html
go to the directory with the scala code, then type:
sbt
The first time this step will take a while. Then you enter the sbt shell, you can exit it by typing 'exit'. Compile the code by typing 'compile'

Once the code successfully compiled, run the code like this and select the first main class (influenza.Prevaccine)
Note: something's wrong here, haven't figured out what yet
run --seed=10 --steps=10 --flutescript=../FluTE-bandits/scripts/bandits/pr
e-vaccination.sh --workdir=../FluTE-bandits/workdir --algo=eps-greedy(0.8) --algo-output=scala-output.csv


## Python bandits (our code)
General idea: FluTE will simulate the environment, given the chosen parameters. It will return the attack rate, which in turn the scala code uses as a reward function for its reinforcement learning task. The lower the attack rate, the higher the reward. So our python code will do something like:
Choose parameters --> run the FluTE script (pre-vaccination.sh) --> get reward --> learn and repeat

Maybe useful: https://stackoverflow.com/questions/3777301/how-to-call-a-shell-script-from-python-code
