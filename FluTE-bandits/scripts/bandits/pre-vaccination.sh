work_dir=$1
flute_rl_dir=$2
flute_rl_bin=$3
config_template=$4
saving_dir=$5
seed=$6
r0=$7
run_length=$8
geo=$9
doses=${10}
quarantine=${11}
priorities=${12}
priorities_no_commas=`echo ${priorities} | sed 's/,//g'`

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${work_dir}

#create config file
python3 $(dirname $PWD)/${flute_rl_dir}/configs/mako-render.py $(dirname $PWD)/rl_dir/${config_template} --label="l0" --seed=${seed} --R0=${r0} --run_length=${run_length} --data_file=$(dirname $PWD)/data/${geo}/${geo} --doses=${doses} --vaccine_priorities=${priorities} --quarantine=${quarantine}> pre.config

#run flute
$(dirname $PWD)/${flute_rl_bin} pre.config > out 2> err

#parse attack rate
ar=`python3 ${script_dir}/parse_attack_rate.py Summary0`
mkdir -p $(dirname $PWD)/${saving_dir}/${priorities_no_commas}
echo $ar >> $(dirname $PWD)/${saving_dir}/${priorities_no_commas}/outcome.txt
