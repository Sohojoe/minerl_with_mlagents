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

# Install custom ml agents

```
cd ml-agents-envs
pip install -e .
cd ..
cd mlagents
pip install -e .
cd ..
```
