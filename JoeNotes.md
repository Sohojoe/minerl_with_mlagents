
# Setup environment

Note - edit between tensorflow and tensorflow-gpu

`conda env create -f environment.yml`

``` bash
conda activate minerl_challenge
pip install -r requirements.txt
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
