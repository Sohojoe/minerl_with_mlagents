import atexit
import glob
import logging
import numpy as np
import os
import subprocess
from typing import *

from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.timers import timed, hierarchical_timer
from mlagents.envs import AllBrainInfo, BrainInfo, BrainParameters
from mlagents.envs import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityActionException,
    UnityTimeOutException,
)
import gym
import minerl
from minerl_to_mlagent_wrapper import MineRLToMLAgentWrapper
from sohojoe_wrappers import (
    KeyboardControlWrapper, PruneActionsWrapper, PruneVisualObservationsWrapper,
    VisualObsAsFloatWrapper
)
from sys import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlagents.envs")


class MineRLUnityEnvironment(BaseUnityEnvironment):
    SCALAR_ACTION_TYPES = (int, np.int32, np.int64, float, np.float32, np.float64)
    SINGLE_BRAIN_ACTION_TYPES = SCALAR_ACTION_TYPES + (list, np.ndarray)
    SINGLE_BRAIN_TEXT_TYPES = (str, list, np.ndarray)

    def __init__(
        self,
        seeds: [int],
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: int = 9001,
        docker_training: bool = False,
        no_graphics: bool = False,
        # timeout_wait: int = 30,
        # args: list = [],
        num_envs: int = 1
    ):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        :string file_name: Name of Unity environment binary.
        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        :bool docker_training: Informs this class whether the process is being run within a container.
        :bool no_graphics: Whether to run the Unity simulator in no-graphics mode
        :int timeout_wait: Time (in seconds) to wait for connection from environment.
        :bool train_mode: Whether to run in training mode, speeding up the simulation, by default.
        :list args: Addition Unity command line arguments
        """

        atexit.register(self._close)
        # self.port = base_port + worker_id
        self._buffer_size = 12000
        # self._version_ = "API-9"
        self._loaded = (
            False
        )  # If true, this means the environment was successfully loaded
        self.proc1 = (
            None
        )  # The process that is started. If None, no process was started
        # self.communicator = self.get_communicator(worker_id, base_port, timeout_wait)
        
        self._worker_id = worker_id

        from minerl.env.malmo import InstanceManager
        self._envs: Dict[str, [Env]] = {}
        self._agent_ids: Dict[str, [str]] = {}
        self._n_agents: Dict[str, int] = {}
        self._global_done: Optional[bool] = None
        self._academy_name = 'MineRLUnityAcademy'
        self._log_path = 'log_path'
        self._brains: Dict[str, BrainParameters] = {}
        self._brain_names: List[str] = []
        self._external_brain_names: List[str] = []
        InstanceManager.configure_malmo_base_port(base_port+worker_id)
        
        for i in range(num_envs):
            print ('InstanceManager:', InstanceManager)
            print ('.MAXINSTANCES', InstanceManager.MAXINSTANCES)
            print ('.REMOTE', InstanceManager.is_remote())
            print ('.ninstances', InstanceManager.ninstances)
            print ('.DEFAULT_IP', InstanceManager.DEFAULT_IP)
            env = gym.make(file_name)
            env = MineRLToMLAgentWrapper(env, seeds[i])
            if self.worker_id is 0:
                env = KeyboardControlWrapper(env)
            env = PruneActionsWrapper(env, [
                # 'attack_jump'
                # ,'camera_left_right'
                'camera_up_down'
                # ,'forward_back'
                ,'left_right'
                ,'place'
                ,'sneak_sprint'
            ])
            # env = PruneVisualObservationsWrapper(env)
            env = VisualObsAsFloatWrapper(env)

            MineRLToMLAgentWrapper.set_wrappers_for_pretraining(file_name, env)

            brain_name = env.brain_parameters.brain_name
            if brain_name not in self._agent_ids:
                self._envs[brain_name] = []
                self._agent_ids[brain_name] = []
                self._n_agents[brain_name] = 0
                self._brain_names.append(brain_name)
                brain = env.brain_parameters
                self._external_brain_names.append(brain_name)
                self._brains[brain_name] = brain
            self._envs[brain_name].append(env)
            self._agent_ids[brain_name].append(env.agent_id)
            self._n_agents[brain_name] += 1
        self._loaded = True

        self._num_brains = len(self._brain_names)
        self._num_external_brains = len(self._external_brain_names)
        # self._resetParameters = dict(aca_params.environment_parameters.float_parameters)
        self._resetParameters = dict()
        logger.info(
            "\n'{0}' started successfully!\n{1}".format(self._academy_name, str(self))
        )
        if self._num_external_brains == 0:
            logger.warning(
                " No Learning Brains set to train found in the Unity Environment. "
                "You will not be able to pass actions to your agent(s)."
            )

    @property
    def logfile_path(self):
        return self._log_path

    @property
    def brains(self):
        return self._brains

    @property
    def global_done(self):
        return self._global_done

    @property
    def academy_name(self):
        return self._academy_name

    @property
    def number_brains(self):
        return self._num_brains

    @property
    def number_external_brains(self):
        return self._num_external_brains

    @property
    def brain_names(self):
        return self._brain_names

    @property
    def external_brain_names(self):
        return self._external_brain_names

    @staticmethod
    def get_communicator(worker_id, base_port, timeout_wait):
        # return RpcCommunicator(worker_id, base_port, timeout_wait)
        return None

    @property
    def worker_id(self):
        return self._worker_id

    @property
    def external_brains(self):
        external_brains = {}
        for brain_name in self.external_brain_names:
            external_brains[brain_name] = self.brains[brain_name]
        return external_brains

    @property
    def reset_parameters(self):
        return self._resetParameters

    def __str__(self):
        return (
            """Unity Academy name: {0}
        Number of Brains: {1}
        Number of Training Brains : {2}
        Reset Parameters :\n\t\t{3}""".format(
                self._academy_name,
                str(self._num_brains),
                str(self._num_external_brains),
                "\n\t\t".join(
                    [
                        str(k) + " -> " + str(self._resetParameters[k])
                        for k in self._resetParameters
                    ]
                ),
            )
            + "\n"
            + "\n".join([str(self._brains[b]) for b in self._brains])
        )

    def reset(
        self, config=None, train_mode=True, custom_reset_parameters=None
    ) -> AllBrainInfo:

        if self._loaded:
            self._global_done = False
            all_brain_info = dict()
            for brain_name in self._external_brain_names:
                for i, env in enumerate(self._envs[brain_name]): #enumerate(xs)
                    brain_info = env.reset()
                    if brain_name not in all_brain_info:
                        all_brain_info[brain_name] = brain_info
                    else:
                        all_brain_info[brain_name] = self._combine_brain_infos([all_brain_info[brain_name], brain_info])
            return all_brain_info
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _combine_brain_infos(self, brain_infos:[BrainInfo])->BrainInfo:
        brain_info = None
        for b in brain_infos:
            if brain_info is None:
                brain_info = b
                continue
            brain_info.agents.append(b.agents[0])
            brain_info.rewards.append(b.rewards[0])
            brain_info.local_done.append(b.local_done[0])
            brain_info.max_reached.append(b.max_reached[0])
            # if (len(brain_info.visual_observations)>0):
            #     brain_info.visual_observations.append(b.visual_observations[0])
            if (len(brain_info.text_observations)>0):
                brain_info.text_observations.append(b.text_observations[0])
            if (len(brain_info.memories)>0):
                brain_info.memories.append(b.memories[0])
            if (len(brain_info.previous_text_actions)>0):
                brain_info.previous_text_actions.append(b.previous_text_actions[0])
            if (len(brain_info.custom_observations)>0):
                brain_info.custom_observations.append(b.custom_observations[0])

            brain_info.action_masks = np.concatenate([brain_info.action_masks, b.action_masks])
            brain_info.visual_observations[0] = np.concatenate([brain_info.visual_observations[0], b.visual_observations[0]])
            brain_info.vector_observations = np.concatenate([brain_info.vector_observations, b.vector_observations])
            brain_info.memories = np.concatenate([brain_info.memories, b.memories])
            # 
            brain_info.previous_vector_actions = np.concatenate([brain_info.previous_vector_actions, b.previous_vector_actions])
        return brain_info

    @timed
    def step(
        self,
        vector_action=None,
        memory=None,
        text_action=None,
        value=None,
        custom_action=None,
    ) -> AllBrainInfo:
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly,
        and returns observation, state, and reward information to the agent.
        :param value: Value estimates provided by agents.
        :param vector_action: Agent's vector action. Can be a scalar or vector of int/floats.
        :param memory: Vector corresponding to memory used for recurrent policies.
        :param text_action: Text action to send to environment for.
        :param custom_action: Optional instance of a CustomAction protobuf message.
        :return: AllBrainInfo  : A Data structure corresponding to the new state of the environment.
        """
        vector_action = {} if vector_action is None else vector_action
        memory = {} if memory is None else memory
        text_action = {} if text_action is None else text_action
        value = {} if value is None else value
        custom_action = {} if custom_action is None else custom_action

        # Check that environment is loaded, and episode is currently running.
        if not self._loaded:
            raise UnityEnvironmentException("No Unity environment is loaded.")
        elif self._global_done:
            raise UnityActionException(
                "The episode is completed. Reset the environment with 'reset()'"
            )
        elif self.global_done is None:
            raise UnityActionException(
                "You cannot conduct step without first calling reset. "
                "Reset the environment with 'reset()'"
            )
        else:
            # vector_action = {self._external_brain_names[0]: vector_action}
            all_brain_info = dict()
            for brain_name in self._external_brain_names:
                for i, env in enumerate(self._envs[brain_name]): #enumerate(xs)
                    a = vector_action[brain_name][i]
                    a = a.reshape(1,a.shape[0])
                    brain_info = env.step(a)
                    if brain_name not in all_brain_info:
                        all_brain_info[brain_name] = brain_info
                    else:
                        all_brain_info[brain_name] = self._combine_brain_infos([all_brain_info[brain_name], brain_info])
                self._global_done = True if True in brain_info.local_done else self._global_done

            # for _b in self._external_brain_names:
            # for i, _b in enumerate(self._external_brain_names): #enumerate(xs)
            #     brain_info = self._envs[i].step(vector_action)
            #     brain_info.memories = memory[_b]
            #     self._global_done = True if True in brain_info.local_done else self._global_done
            #     all_brain_info[_b]=brain_info
            return all_brain_info

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded:
            self._close()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _close(self):
        self._loaded = False
        # self.communicator.close()
        if self.proc1 is not None:
            self.proc1.kill()

    @classmethod
    def _flatten(cls, arr) -> List[float]:
        """
        Converts arrays to list.
        :param arr: numpy vector.
        :return: flattened list.
        """
        if isinstance(arr, cls.SCALAR_ACTION_TYPES):
            arr = [float(arr)]
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if len(arr) == 0:
            return arr
        if isinstance(arr[0], np.ndarray):
            arr = [item for sublist in arr for item in sublist.tolist()]
        if isinstance(arr[0], list):
            arr = [item for sublist in arr for item in sublist]
        arr = [float(x) for x in arr]
        return arr
