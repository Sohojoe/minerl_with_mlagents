#!/bin/bash
set -e


AICROWD_DATA_ENABLED="YES"
if [[ " $@ " =~ " --no-data " ]]; then
   AICROWD_DATA_ENABLED="NO"
else
    python3 ./utility/verify_or_download_data.py
fi



EXTRAOUTPUT=" > /dev/null 2>&1 "
if [[ " $@ " =~ " --verbose " ]]; then
   EXTRAOUTPUT=""
fi

# Run local name server
eval "pyro4-ns $EXTRAOUTPUT &"
trap "kill -11 $! > /dev/null 2>&1;" EXIT

# Run instance manager to generate performance report
export EVALUATION_STAGE='manager'
eval "python3 run.py --seeds 1 $EXTRAOUTPUT &"
trap "kill -11 $! > /dev/null 2>&1;" EXIT

# Run the training phase
sleep 2
echo "RUNNING TRAINING!"
export MINERL_INSTANCE_MANAGER_REMOTE="1"
export EVALUATION_STAGE='training'
export EVALUATION_RUNNING_ON='local'
export EXITED_SIGNAL_PATH='shared/training_exited'

# echo "** HACK ** install ml-agents"
# pip install git+git://github.com/Sohojoe/ml-agents-envs-python@master
# pip install git+git://github.com/Sohojoe/ml-agents-python@master

echo "** HACK ** joe env vars"
export CUDA_VISIBLE_DEVICES=-1
# export MINERL_GYM_ENV='MineRLTreechop-v0'
export MINERL_GYM_ENV='MineRLObtainDiamond-v0'
export MINERL_DATA_ROOT='./data'
export MINERL_TRAINING_MAX_INSTANCES=5
export MINERL_TRAINING_MAX_STEPS=8000000
export MINERL_TRAINING_TIMEOUT_MINUTES=5760

rm -f $EXITED_SIGNAL_PATH
export ENABLE_AICROWD_JSON_OUTPUT='False'
eval "python3 run.py $EXTRAOUTPUT && touch $EXITED_SIGNAL_PATH || touch $EXITED_SIGNAL_PATH &"
trap "kill -11 $! > /dev/null 2>&1;" EXIT

# View the evaluation state
export ENABLE_AICROWD_JSON_OUTPUT='True'
python3 utility/parser.py || true
kill $(jobs -p)
