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
from sohojoe_wrappers import KeyboardControlWrapper, PruneActionsWrapper

# from mlagents.env.communicator_objects import (
#     UnityRLInput,
#     UnityRLOutput,
#     AgentActionProto,
#     EnvironmentParametersProto,
#     UnityRLInitializationInput,
#     UnityRLInitializationOutput,
#     UnityInput,
#     UnityOutput,
#     CustomResetParameters,
#     CustomAction,
# )

# from .rpc_communicator import RpcCommunicator
from sys import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlagents.envs")


class MineRLUnityEnvironment(BaseUnityEnvironment):
    SCALAR_ACTION_TYPES = (int, np.int32, np.int64, float, np.float32, np.float64)
    SINGLE_BRAIN_ACTION_TYPES = SCALAR_ACTION_TYPES + (list, np.ndarray)
    SINGLE_BRAIN_TEXT_TYPES = (str, list, np.ndarray)

    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: int = 5005,
        seed: int = 0,
        docker_training: bool = False,
        no_graphics: bool = False,
        timeout_wait: int = 30,
        args: list = [],
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
        self.port = base_port + worker_id
        self._buffer_size = 12000
        # self._version_ = "API-9"
        self._loaded = (
            False
        )  # If true, this means the environment was successfully loaded
        self.proc1 = (
            None
        )  # The process that is started. If None, no process was started
        # self.communicator = self.get_communicator(worker_id, base_port, timeout_wait)
        self.worker_id = worker_id

        env = gym.make(file_name)
        env = MineRLToMLAgentWrapper(env, self.port)
        # env = KeyboardControlWrapper(env)
        env = PruneActionsWrapper(env, [
            # 'attack_jump'
            # ,'camera_left_right'
            'camera_up_down'
            # ,'forward_back'
            ,'left_right'
            ,'place'
            ,'sneak_sprint'
        ])
        self._loaded = True

        # rl_init_parameters_in = UnityRLInitializationInput(seed=seed)
        # try:
        #     aca_params = self.send_academy_parameters(rl_init_parameters_in)
        # except UnityTimeOutException:
        #     self._close()
        #     raise
        # TODO : think of a better way to expose the academyParameters
        # self._unity_version = aca_params.version
        # if self._unity_version != self._version_:
        #     self._close()
        #     raise UnityEnvironmentException(
        #         "The API number is not compatible between Unity and python. Python API : {0}, Unity API : "
        #         "{1}.\nPlease go to https://github.com/Unity-Technologies/ml-agents to download the latest version "
        #         "of ML-Agents.".format(self._version_, self._unity_version)
        #     )

        self._envs = [env]
        self._agents: Dict[str, str] = {}
        self._n_agents: Dict[str, int] = {}
        self._agents[env.brain_parameters.brain_name] = env.agent_id
        self._n_agents[env.brain_parameters.brain_name] = 1
        self._global_done: Optional[bool] = None
        self._academy_name = 'MineRLUnityAcademy'
        self._log_path = 'log_path'
        self._brains: Dict[str, BrainParameters] = {}
        self._brain_names: List[str] = []
        self._external_brain_names: List[str] = []
        # for brain_param in aca_params.brain_parameters:
        #     self._brain_names += [brain_param.brain_name]
        #     self._brains[brain_param.brain_name] = BrainParameters.from_proto(
        #         brain_param
        #     )
        #     if brain_param.is_training:
        #         self._external_brain_names += [brain_param.brain_name]
        # brain_name = 'MineRLUnityBrain'
        # self._brain_names.append(brain_name)
        # self._external_brain_names.append(brain_name)
        # self._brains[brain_name] = BrainParameters(
        #     brain_name=brain_name,
        #     vector_observation_space_size=3,
        #     num_stacked_vector_observations=1,
        #     camera_resolutions=[], #List[Dict],
        #     vector_action_space_size=[], #List[int],
        #     vector_action_descriptions=[], #List[str],
        #     vector_action_space_type=0 #: int,
        # )
        self._envs = [env]
        for e in self._envs:
            brain = e.brain_parameters
            self._brain_names.append(brain.brain_name)
            self._external_brain_names.append(brain.brain_name)
            self._brains[brain.brain_name] = brain
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
    def external_brains(self):
        external_brains = {}
        for brain_name in self.external_brain_names:
            external_brains[brain_name] = self.brains[brain_name]
        return external_brains

    @property
    def reset_parameters(self):
        return self._resetParameters

    # def executable_launcher(self, file_name, docker_training, no_graphics, args):
    #     cwd = os.getcwd()
    #     file_name = (
    #         file_name.strip()
    #         .replace(".app", "")
    #         .replace(".exe", "")
    #         .replace(".x86_64", "")
    #         .replace(".x86", "")
    #     )
    #     true_filename = os.path.basename(os.path.normpath(file_name))
    #     logger.debug("The true file name is {}".format(true_filename))
    #     launch_string = None
    #     if platform == "linux" or platform == "linux2":
    #         candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
    #         if len(candidates) == 0:
    #             candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
    #         if len(candidates) == 0:
    #             candidates = glob.glob(file_name + ".x86_64")
    #         if len(candidates) == 0:
    #             candidates = glob.glob(file_name + ".x86")
    #         if len(candidates) > 0:
    #             launch_string = candidates[0]

    #     elif platform == "darwin":
    #         candidates = glob.glob(
    #             os.path.join(
    #                 cwd, file_name + ".app", "Contents", "MacOS", true_filename
    #             )
    #         )
    #         if len(candidates) == 0:
    #             candidates = glob.glob(
    #                 os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
    #             )
    #         if len(candidates) == 0:
    #             candidates = glob.glob(
    #                 os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
    #             )
    #         if len(candidates) == 0:
    #             candidates = glob.glob(
    #                 os.path.join(file_name + ".app", "Contents", "MacOS", "*")
    #             )
    #         if len(candidates) > 0:
    #             launch_string = candidates[0]
    #     elif platform == "win32":
    #         candidates = glob.glob(os.path.join(cwd, file_name + ".exe"))
    #         if len(candidates) == 0:
    #             candidates = glob.glob(file_name + ".exe")
    #         if len(candidates) > 0:
    #             launch_string = candidates[0]
    #     if launch_string is None:
    #         self._close()
    #         raise UnityEnvironmentException(
    #             "Couldn't launch the {0} environment. "
    #             "Provided filename does not match any environments.".format(
    #                 true_filename
    #             )
    #         )
    #     else:
    #         logger.debug("This is the launch string {}".format(launch_string))
    #         # Launch Unity environment
    #         if not docker_training:
    #             if no_graphics:
    #                 self.proc1 = subprocess.Popen(
    #                     [
    #                         launch_string,
    #                         "-nographics",
    #                         "-batchmode",
    #                         "--port",
    #                         str(self.port),
    #                     ]
    #                     + args
    #                 )
    #             else:
    #                 self.proc1 = subprocess.Popen(
    #                     [launch_string, "--port", str(self.port)] + args
    #                 )
    #         else:
    #             """
    #             Comments for future maintenance:
    #                 xvfb-run is a wrapper around Xvfb, a virtual xserver where all
    #                 rendering is done to virtual memory. It automatically creates a
    #                 new virtual server automatically picking a server number `auto-servernum`.
    #                 The server is passed the arguments using `server-args`, we are telling
    #                 Xvfb to create Screen number 0 with width 640, height 480 and depth 24 bits.
    #                 Note that 640 X 480 are the default width and height. The main reason for
    #                 us to add this is because we'd like to change the depth from the default
    #                 of 8 bits to 24.
    #                 Unfortunately, this means that we will need to pass the arguments through
    #                 a shell which is why we set `shell=True`. Now, this adds its own
    #                 complications. E.g SIGINT can bounce off the shell and not get propagated
    #                 to the child processes. This is why we add `exec`, so that the shell gets
    #                 launched, the arguments are passed to `xvfb-run`. `exec` replaces the shell
    #                 we created with `xvfb`.
    #             """
    #             docker_ls = (
    #                 "exec xvfb-run --auto-servernum"
    #                 " --server-args='-screen 0 640x480x24'"
    #                 " {0} --port {1}"
    #             ).format(launch_string, str(self.port))
    #             self.proc1 = subprocess.Popen(
    #                 docker_ls,
    #                 stdout=subprocess.PIPE,
    #                 stderr=subprocess.PIPE,
    #                 shell=True,
    #             )

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
        # if config is None:
        #     config = self._resetParameters
        # elif config:
        #     logger.info(
        #         "Academy reset with parameters: {0}".format(
        #             ", ".join([str(x) + " -> " + str(config[x]) for x in config])
        #         )
        #     )
        # for k in config:
        #     if (k in self._resetParameters) and (isinstance(config[k], (int, float))):
        #         self._resetParameters[k] = config[k]
        #     elif not isinstance(config[k], (int, float)):
        #         raise UnityEnvironmentException(
        #             "The value for parameter '{0}'' must be an Integer or a Float.".format(
        #                 k
        #             )
        #         )
        #     else:
        #         raise UnityEnvironmentException(
        #             "The parameter '{0}' is not a valid parameter.".format(k)
        #         )

        if self._loaded:
            self._global_done = False
            # outputs = self.communicator.exchange(
            #     self._generate_reset_input(train_mode, config, custom_reset_parameters)
            # )
            # if outputs is None:
            #     raise UnityCommunicationException("Communicator has stopped.")
            # rl_output = outputs.rl_output
            # s = self._get_state(rl_output)
            # self._global_done = s[1]
            # for _b in self._external_brain_names:
            #     self._n_agents[_b] = len(s[0][_b].agents)
            # return s[0]
            all_brain_info = dict()
            # for _b in self._external_brain_names:
            for i, _b in enumerate(self._external_brain_names): #enumerate(xs)
                brain_info = self._envs[i].reset()
                all_brain_info[_b]=brain_info
            return all_brain_info
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

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
            # for _b in self._external_brain_names:
            for i, _b in enumerate(self._external_brain_names): #enumerate(xs)
                brain_info = self._envs[i].step(vector_action)
                brain_info.memories = memory[_b]
                self._global_done = True if True in brain_info.local_done else self._global_done
                all_brain_info[_b]=brain_info
            return all_brain_info

        # raise UnityActionException("Not Implemented")        
        # else:
        #     if isinstance(vector_action, self.SINGLE_BRAIN_ACTION_TYPES):
        #         if self._num_external_brains == 1:
        #             vector_action = {self._external_brain_names[0]: vector_action}
        #         elif self._num_external_brains > 1:
        #             raise UnityActionException(
        #                 "You have {0} brains, you need to feed a dictionary of brain names a keys, "
        #                 "and vector_actions as values".format(self._num_brains)
        #             )
        #         else:
        #             raise UnityActionException(
        #                 "There are no external brains in the environment, "
        #                 "step cannot take a vector_action input"
        #             )

        #     if isinstance(memory, self.SINGLE_BRAIN_ACTION_TYPES):
        #         if self._num_external_brains == 1:
        #             memory = {self._external_brain_names[0]: memory}
        #         elif self._num_external_brains > 1:
        #             raise UnityActionException(
        #                 "You have {0} brains, you need to feed a dictionary of brain names as keys "
        #                 "and memories as values".format(self._num_brains)
        #             )
        #         else:
        #             raise UnityActionException(
        #                 "There are no external brains in the environment, "
        #                 "step cannot take a memory input"
        #             )

        #     if isinstance(text_action, self.SINGLE_BRAIN_TEXT_TYPES):
        #         if self._num_external_brains == 1:
        #             text_action = {self._external_brain_names[0]: text_action}
        #         elif self._num_external_brains > 1:
        #             raise UnityActionException(
        #                 "You have {0} brains, you need to feed a dictionary of brain names as keys "
        #                 "and text_actions as values".format(self._num_brains)
        #             )
        #         else:
        #             raise UnityActionException(
        #                 "There are no external brains in the environment, "
        #                 "step cannot take a value input"
        #             )

        #     if isinstance(value, self.SINGLE_BRAIN_ACTION_TYPES):
        #         if self._num_external_brains == 1:
        #             value = {self._external_brain_names[0]: value}
        #         elif self._num_external_brains > 1:
        #             raise UnityActionException(
        #                 "You have {0} brains, you need to feed a dictionary of brain names as keys "
        #                 "and state/action value estimates as values".format(
        #                     self._num_brains
        #                 )
        #             )
        #         else:
        #             raise UnityActionException(
        #                 "There are no external brains in the environment, "
        #                 "step cannot take a value input"
        #             )

        #     if isinstance(custom_action, CustomAction):
        #         if self._num_external_brains == 1:
        #             custom_action = {self._external_brain_names[0]: custom_action}
        #         elif self._num_external_brains > 1:
        #             raise UnityActionException(
        #                 "You have {0} brains, you need to feed a dictionary of brain names as keys "
        #                 "and CustomAction instances as values".format(self._num_brains)
        #             )
        #         else:
        #             raise UnityActionException(
        #                 "There are no external brains in the environment, "
        #                 "step cannot take a custom_action input"
        #             )

        #     for brain_name in (
        #         list(vector_action.keys())
        #         + list(memory.keys())
        #         + list(text_action.keys())
        #     ):
        #         if brain_name not in self._external_brain_names:
        #             raise UnityActionException(
        #                 "The name {0} does not correspond to an external brain "
        #                 "in the environment".format(brain_name)
        #             )

        #     for brain_name in self._external_brain_names:
        #         n_agent = self._n_agents[brain_name]
        #         if brain_name not in vector_action:
        #             if self._brains[brain_name].vector_action_space_type == "discrete":
        #                 vector_action[brain_name] = (
        #                     [0.0]
        #                     * n_agent
        #                     * len(self._brains[brain_name].vector_action_space_size)
        #                 )
        #             else:
        #                 vector_action[brain_name] = (
        #                     [0.0]
        #                     * n_agent
        #                     * self._brains[brain_name].vector_action_space_size[0]
        #                 )
        #         else:
        #             vector_action[brain_name] = self._flatten(vector_action[brain_name])
        #         if brain_name not in memory:
        #             memory[brain_name] = []
        #         else:
        #             if memory[brain_name] is None:
        #                 memory[brain_name] = []
        #             else:
        #                 memory[brain_name] = self._flatten(memory[brain_name])
        #         if brain_name not in text_action:
        #             text_action[brain_name] = [""] * n_agent
        #         else:
        #             if text_action[brain_name] is None:
        #                 text_action[brain_name] = [""] * n_agent
        #             if isinstance(text_action[brain_name], str):
        #                 text_action[brain_name] = [text_action[brain_name]] * n_agent
        #         if brain_name not in custom_action:
        #             custom_action[brain_name] = [None] * n_agent
        #         else:
        #             if custom_action[brain_name] is None:
        #                 custom_action[brain_name] = [None] * n_agent
        #             if isinstance(custom_action[brain_name], CustomAction):
        #                 custom_action[brain_name] = [
        #                     custom_action[brain_name]
        #                 ] * n_agent

        #         number_text_actions = len(text_action[brain_name])
        #         if not ((number_text_actions == n_agent) or number_text_actions == 0):
        #             raise UnityActionException(
        #                 "There was a mismatch between the provided text_action and "
        #                 "the environment's expectation: "
        #                 "The brain {0} expected {1} text_action but was given {2}".format(
        #                     brain_name, n_agent, number_text_actions
        #                 )
        #             )

        #         discrete_check = (
        #             self._brains[brain_name].vector_action_space_type == "discrete"
        #         )

        #         expected_discrete_size = n_agent * len(
        #             self._brains[brain_name].vector_action_space_size
        #         )

        #         continuous_check = (
        #             self._brains[brain_name].vector_action_space_type == "continuous"
        #         )

        #         expected_continuous_size = (
        #             self._brains[brain_name].vector_action_space_size[0] * n_agent
        #         )

        #         if not (
        #             (
        #                 discrete_check
        #                 and len(vector_action[brain_name]) == expected_discrete_size
        #             )
        #             or (
        #                 continuous_check
        #                 and len(vector_action[brain_name]) == expected_continuous_size
        #             )
        #         ):
        #             raise UnityActionException(
        #                 "There was a mismatch between the provided action and "
        #                 "the environment's expectation: "
        #                 "The brain {0} expected {1} {2} action(s), but was provided: {3}".format(
        #                     brain_name,
        #                     str(expected_discrete_size)
        #                     if discrete_check
        #                     else str(expected_continuous_size),
        #                     self._brains[brain_name].vector_action_space_type,
        #                     str(vector_action[brain_name]),
        #                 )
        #             )

        #     step_input = self._generate_step_input(
        #         vector_action, memory, text_action, value, custom_action
        #     )
        #     with hierarchical_timer("communicator.exchange"):
        #         outputs = self.communicator.exchange(step_input)
        #     if outputs is None:
        #         raise UnityCommunicationException("Communicator has stopped.")
        #     rl_output = outputs.rl_output
        #     state = self._get_state(rl_output)
        #     self._global_done = state[1]
        #     for _b in self._external_brain_names:
        #         self._n_agents[_b] = len(state[0][_b].agents)
        #     return state[0]

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

    # def _get_state(self, output: UnityRLOutput) -> Tuple[AllBrainInfo, bool]:
    #     """
    #     Collects experience information from all external brains in environment at current step.
    #     :return: a dictionary of BrainInfo objects.
    #     """
    #     _data = {}
    #     global_done = output.global_done
    #     for brain_name in output.agentInfos:
    #         agent_info_list = output.agentInfos[brain_name].value
    #         _data[brain_name] = BrainInfo.from_agent_proto(
    #             self.worker_id, agent_info_list, self.brains[brain_name]
    #         )
    #     return _data, global_done

    # @timed
    # def _generate_step_input(
    #     self, vector_action, memory, text_action, value, custom_action
    # ) -> UnityInput:
    #     rl_in = UnityRLInput()
    #     for b in vector_action:
    #         n_agents = self._n_agents[b]
    #         if n_agents == 0:
    #             continue
    #         _a_s = len(vector_action[b]) // n_agents
    #         _m_s = len(memory[b]) // n_agents
    #         for i in range(n_agents):
    #             action = AgentActionProto(
    #                 vector_actions=vector_action[b][i * _a_s : (i + 1) * _a_s],
    #                 memories=memory[b][i * _m_s : (i + 1) * _m_s],
    #                 text_actions=text_action[b][i],
    #                 custom_action=custom_action[b][i],
    #             )
    #             if b in value:
    #                 if value[b] is not None:
    #                     action.value = float(value[b][i])
    #             rl_in.agent_actions[b].value.extend([action])
    #             rl_in.command = 0
    #     return self.wrap_unity_input(rl_in)

    # def _generate_reset_input(
    #     self, training, config, custom_reset_parameters
    # ) -> UnityInput:
    #     rl_in = UnityRLInput()
    #     rl_in.is_training = training
    #     rl_in.environment_parameters.CopyFrom(EnvironmentParametersProto())
    #     for key in config:
    #         rl_in.environment_parameters.float_parameters[key] = config[key]
    #     if custom_reset_parameters is not None:
    #         rl_in.environment_parameters.custom_reset_parameters.CopyFrom(
    #             custom_reset_parameters
    #         )
    #     rl_in.command = 1
    #     return self.wrap_unity_input(rl_in)

    # def send_academy_parameters(
    #     self, init_parameters: UnityRLInitializationInput
    # ) -> UnityRLInitializationOutput:
    #     inputs = UnityInput()
    #     inputs.rl_initialization_input.CopyFrom(init_parameters)
    #     return self.communicator.initialize(inputs).rl_initialization_output

    # @staticmethod
    # def wrap_unity_input(rl_input: UnityRLInput) -> UnityInput:
    #     result = UnityInput()
    #     result.rl_input.CopyFrom(rl_input)
    #     return result
