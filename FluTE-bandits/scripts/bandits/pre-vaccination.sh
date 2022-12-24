work_dir=$1
flute_rl_dir=$2
flute_rl_bin=$3
config_template=$4
seed=$5
r0=$6
run_length=$7
geo=$8
doses=$9
priorities=${10}
priorities_no_commas=`echo ${priorities} | sed 's/,//g'`

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "workdir:${workdir}"
echo "flute_rl_dir:${flute_rl_dir}"
echo "flute_rl_bin:${flute_rl_bin}"
echo "config_template:${config_template}"
echo "seed:${seed}"
echo "r0:${r0}"
echo "run_length:${run_length}"
echo "geo:${geo}"
echo "doses:${doses}"
echo "priorities:${priorities}"

cd ${work_dir}

echo "Creating config file..."
#create config file
python $(dirname $PWD)/${flute_rl_dir}/configs/mako-render.py $(dirname $PWD)/rl_dir/${config_template} --label="l0" --seed=${seed} --R0=${r0} --run_length=${run_length} --data_file=$(dirname $PWD)/data/${geo}/${geo} --doses=${doses} --vaccine_priorities=${priorities} > pre.config
echo "Created config file."

echo "Running FluTE..."
#run flute
$(dirname $PWD)/${flute_rl_bin} pre.config > out 2> err
echo "Done running FluTE."

echo "Parsing attack rate..."
#parse attack rate
ar=`python ${script_dir}/parse_attack_rate.py Summary0`
echo $ar
