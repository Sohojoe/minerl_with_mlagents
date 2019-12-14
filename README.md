# MineRL+ML-Agents

I'm open-sourcing my code for the [mine_rl competition](https://github.com/minerllabs/minerl) for NeurIPS 2019.

My main contribition is the wrapper for the Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents) python code, but note that I never got good results so either there are some bugs or tweaks to the hyperparams are needed.


# Setup environment

Note - edit between tensorflow and tensorflow-gpu

``` bash
conda env create -f environment.yml
conda activate minerl_challenge
pip install -r requirements.txt
python ./utility/verify_or_download_data.py
```

## Install custom ml agents

``` bash
cd ml-agents-envs
pip install -e .
cd ..
cd mlagents
pip install -e .
cd ..
```

# update ml-agents

## notes

* root is mlagents (not ml-agents)
* copy ml-agents\mlagents to mlagents
  * copy setup.py
  * edit setup.py if needed
* copy ml-agents-env
* copy gym-unity

## mlagents/trainers/components/reward_signals/gail/signal.py

replace line (12)

* from mlagents.trainers.demo_loader import demo_to_buffer

with

* from minerl_demo_to_buffer import demo_to_buffer

## mlagents/trainers/learn.py

replace line (182)

* `model_path = "./models/{run_id}-{sub_id}".format(`

to

* `model_path = "./train/{run_id}-{sub_id}".format(`

