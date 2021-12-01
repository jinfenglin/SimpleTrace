#!/bin/bash
#$ -q long     # Specify queue (use ‘debug’ for development)
#$ -N git_split_data      # Specify job name
if [ -r /opt/crc/Modules/current/init/bash ]; then
    source /opt/crc/Modules/current/init/bash
fi
module load python/3.7.3
cd /afs/crc.nd.edu/user/a/apoudel/projects/MLPROJECT
source ./venv/bin/activate
# $data_dir=/scratch365/jlin6/data/git_data/clean/run_2
# $out_dir=~/projects/SEBert/git_split_data_final/
# $py_file=./scripts/data_process/split_dataset.py
python ./data_process/train.py 