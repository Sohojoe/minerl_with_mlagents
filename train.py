# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

import coloredlogs
# coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

from typing import Any, Callable, Dict, Optional
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
import numpy as np
from minerl_unity_environment import MineRLUnityEnvironment
from mlagents.envs import BrainParameters

def create_environment_factory(
    env_path: str,
    docker_target_name: str,
    no_graphics: bool,
    seed: Optional[int],
    start_port: int,
) -> Callable[[int], BaseUnityEnvironment]:
    # docker_training = docker_target_name is not None
    # if docker_training and env_path is not None:
    #     """
    #         Comments for future maintenance:
    #             Some OS/VM instances (e.g. COS GCP Image) mount filesystems
    #             with COS flag which prevents execution of the Unity scene,
    #             to get around this, we will copy the executable into the
    #             container.
    #         """
    #     # Navigate in docker path and find env_path and copy it.
    #     env_path = prepare_for_docker_run(docker_target_name, env_path)
    docker_training = None
    seed_count = 10000
    seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

    # def create_unity_environment(worker_id: int) -> UnityEnvironment:
    # def create_unity_environment(num_envs: int):
    def create_unity_environment(worker_id: int):
        num_envs=1
        seeds = [np.random.randint(0, seed_count) for _ in range(num_envs)]
        if seed:
            seeds[0] = seed
        env = MineRLUnityEnvironment(
            file_name=env_path,
            num_envs=num_envs,
            worker_id=worker_id,
            seeds=seeds,
            docker_training=docker_training,
            no_graphics=no_graphics,
        )
        return env

    return create_unity_environment



def main():
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    # data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)
    # MINERL_GYM_ENV = 'MineRLTreechop-v0'
    # MINERL_GYM_ENV = 'MineRLNavigate-v0'
    MINERL_GYM_ENV = 'MineRLNavigateDense-v0'
    # MINERL_GYM_ENV = 'MineRLNavigateExtreme-v0'
    # MINERL_GYM_ENV = 'MineRLNavigateExtremeDense-v0'
    # MINERL_GYM_ENV = 'MineRLObtainIronPickaxe-v0'
    # MINERL_GYM_ENV = 'MineRLObtainIronPickaxeDense-v0'
    # MINERL_GYM_ENV = 'MineRLObtainDiamond-v0'
    # MINERL_GYM_ENV = 'MineRLObtainDiamondDense-v0'
    #                         # for debug use
    # MINERL_GYM_ENV = 'MineRLNavigateDenseFixed-v0'
    # MINERL_GYM_ENV = 'MineRLObtainTest-v0'

    os.environ['MINERL_DATA_ROOT']=MINERL_DATA_ROOT

    from trainer_mlagents import main as unity_main
    import sys
    argv = sys.argv[1:]
    argv.append('config/mlagents_gail_config.yaml')
    argv.append('--train')
    argv.append('--env='+MINERL_GYM_ENV)
    # argv.append('--num-envs=2')
    # argv.append('--num-envs=5')
    argv.append('--run-id=MineRLNavigateDense-018')

    # env = MineRLUnityEnvironment(MINERL_GYM_ENV)
    from minerl.env.malmo import InstanceManager
    InstanceManager.MAXINSTANCES = MINERL_TRAINING_MAX_INSTANCES
    # InstanceManager.allocate_pool(5)

    unity_main(argv, create_environment_factory)
    # gym.envs.registry.env_specs[MINERL_GYM_ENV]

    # Sample code for illustration, add your training code below
    # env = gym.make(MINERL_GYM_ENV)
    # print ('action_space:', env.action_space)
    # print ('observation_space:', env.observation_space)
    # print ('reward_range:', env.reward_range)
    # print ('metadata:', env.metadata)
    # print ('-----')

#     actions = [env.action_space.sample() for _ in range(10)] # Just doing 10 samples in this example
#     xposes = []
#     for _ in range(1):
#         obs = env.reset()
#         done = False
#         netr = 0

#         # Limiting our code to 1024 steps in this example, you can do "while not done" to run till end
#         while not done:

            # To get better view in your training phase, it is suggested
            # to register progress continuously, example when 54% completed
            # aicrowd_helper.register_progress(0.54)

            # To fetch latest information from instance manager, you can run below when you want to know the state
            #>> parser.update_information()
            #>> print(parser.payload)
            # .payload: provide AIcrowd generated json
            # Example: {'state': 'RUNNING', 'score': {'score': 0.0, 'score_secondary': 0.0}, 'instances':
            #  {'1': {'totalNumberSteps': 2001, 'totalNumberEpisodes': 0, 'currentEnvironment': 'MineRLObtainDiamond-v0',
            #  'state': 'IN_PROGRESS', 'episodes': [{'numTicks': 2001, 'environment': 'MineRLObtainDiamond-v0',
            #  'rewards': 0.0, 'state': 'IN_PROGRESS'}], 'score': {'score': 0.0, 'score_secondary': 0.0}}}}
            # .current_state: provide indepth state information avaiable as dictionary (key: instance id)

    # Save trained model to train/ directory
    # Training 100% Completed
    # aicrowd_helper.register_progress(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    main()
