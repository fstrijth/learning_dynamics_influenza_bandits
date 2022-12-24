# learning_dynamics_influenza_bandits


## FluTE
Used this FluTE implementation https://github.com/vub-ai-lab/FluTE-bandits. Apart from changing the starting scripts around a bit, the code remains the same

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
TODO
