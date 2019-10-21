import gym
from mlagents.envs import BrainInfo, BrainParameters
import minerl
import numpy as np
from typing import Any, Callable, Dict, Generator, List, TypeVar
from collections import OrderedDict


class MineRLToMLAgentWrapper(gym.Wrapper):
    """A class to wrap a MineRL OpenAI.gym environment as a UnityMLAgent environment.

    Specifically, maps observations, actions, rewards, so that we can train with ml-agents
    """
    # static 
    static_brain_params:Dict[str, BrainParameters] = dict()
    static_wrapped_env: Dict[str, gym.Wrapper] = dict()

    def __init__(self, env, port):
        super(MineRLToMLAgentWrapper, self).__init__(env)   
        env_id = env.spec.id
        self._is_processing_obs = False
        self._hack_always_chop = False

        self._minerl_action_space = gym.envs.registry.env_specs[env_id]._kwargs['action_space'].spaces
        self._minerl_observation_space = gym.envs.registry.env_specs[env_id]._kwargs['observation_space'].spaces
        self._mlagent_action_space = dict()
        if 'forward' and 'back' in self._minerl_action_space:
            self._mlagent_action_space['forward_back']=['noop','forward','back']
        if 'left' and 'right' in self._minerl_action_space:
            self._mlagent_action_space['left_right']=['noop','left','right']
        if self._hack_always_chop:
            if 'jump' in self._minerl_action_space:
                self._mlagent_action_space['jump']=['noop','jump']
        else:
            if 'attack' and 'jump' in self._minerl_action_space:
                self._mlagent_action_space['attack_jump']=['noop','attack','jump']
        self._mlagent_action_space['camera_left_right']=['noop','camera_left','camera_right']
        self._mlagent_action_space['camera_up_down']=['noop','camera_up','camera_down']
        if 'sneak' and 'sprint' in self._minerl_action_space:
            self._mlagent_action_space['sneak_sprint']=['noop','sneak','sprint']
        for key in ['craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']:
            if key in self._minerl_action_space: 
                self._mlagent_action_space[key]=[x for x in self._minerl_action_space[key].values]

        brain_name = env_id
        self._agent_id = brain_name+'-'+str(port)
        vector_observation_space_size = int(0)
        self._vector_obs_keys: Dict[str, float] = dict()
        if 'equipped_items' in self._minerl_observation_space:
            for k,v in self._minerl_observation_space['equipped_items'].spaces['mainhand'].spaces.items():
                self._vector_obs_keys['equipped_items.'+k]=vector_observation_space_size
                if type(v) is minerl.env.spaces.Box:
                    vector_observation_space_size += 1
                elif type(v) is minerl.env.spaces.Enum:
                    vector_observation_space_size += len(v)
                else:
                    raise NotImplementedError
        # Box()
        for obs in ['compassAngle']:
            if obs in self._minerl_observation_space:
                self._vector_obs_keys[obs]=vector_observation_space_size
                vector_observation_space_size += 1
        # Dict()
        for obs in ['inventory']:
            if obs in self._minerl_observation_space:
                self._vector_obs_keys[obs]=vector_observation_space_size
                for k,v in self._minerl_observation_space[obs].spaces.items():
                    if type(v) is minerl.env.spaces.Box:
                        vector_observation_space_size += 1
                    else:
                        raise NotImplementedError
        num_stacked_vector_observations = int(1)
        camera_resolutions = [{
            "width": self._minerl_observation_space['pov'].shape[0], 
            "height": self._minerl_observation_space['pov'].shape[1], 
            "blackAndWhite": self._minerl_observation_space['pov'].shape[2] == 0
            }]
        vector_action_space_size = [len(v) for k,v in self._mlagent_action_space.items()]
        vector_action_descriptions = [
            # ".".join(k, i)
            # for i in v
            k
            for k,v in self._mlagent_action_space.items()
            ]
        vector_action_space_type = 0  #'discrete'
        self._brain_params = BrainParameters(
            brain_name, 
            vector_observation_space_size, 
            num_stacked_vector_observations, 
            camera_resolutions, 
            vector_action_space_size, 
            vector_action_descriptions, 
            vector_action_space_type
            )
        if brain_name not in MineRLToMLAgentWrapper.static_brain_params:
            MineRLToMLAgentWrapper.static_brain_params[brain_name] = self._brain_params

    def _process_action(self, raw_action_in):
        # map mlagent action to minerl
        # map mlagent action to minerl
        raw_action_in = raw_action_in[self.env.spec.id]
        action_in = raw_action_in[0]
        action = self.action_space.sample()
        for act_k in action:
        # for mine_k, mine_v in self._minerl_action_space.items():
            act_v = action[act_k]
            try:
                for i,k in enumerate(act_v):
                    act_v[i] = -1
            except TypeError:
                act_v = -1
            actions = ['forward', 'back', 'left', 'right', 'sneak', 'sprint', 'jump', 'attack']
            if self._hack_always_chop:
                actions = ['forward', 'back', 'left', 'right', 'sneak', 'sprint', 'jump']
                if act_k in ['attack']:
                    act_v = type(act_v) (1)
            if act_k in actions:
                for i, v in enumerate(self._mlagent_action_space.items()): 
                    if act_k in v[0] and act_k in v[1]:
                        act_v = type(act_v) (action_in[i] == v[1].index(act_k))
            if act_k in ['craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']:
                for i, v in enumerate(self._mlagent_action_space.items()): 
                    if act_k in v[0] and act_k:
                        act_v = type(act_v) (action_in[i])
            if act_k == 'camera':
                # self._mlagent_action_space['camera_left_right']=['noop','camera_left','camera_right']
                # self._mlagent_action_space['camera_up_down']=['noop','camera_up','camera_down']
                # VIEW_STEP=90
                VIEW_STEP=9
                i = list(self._mlagent_action_space.keys()).index('camera_left_right')
                v = action_in[i]
                act_v[1] = -VIEW_STEP if v == 1 else VIEW_STEP if v == 2 else 0
                i = list(self._mlagent_action_space.keys()).index('camera_up_down')
                v = action_in[i]
                act_v[0] = -VIEW_STEP if v == 1 else VIEW_STEP if v == 2 else 0
            # print(act_k, act_v)
            try:
                for i,k in enumerate(act_v):
                    if k == -1:
                        raise NotImplementedError('_process_action key error '+ act_k)
            except TypeError:
                if act_v == -1:
                    raise NotImplementedError('_process_action key error '+ act_k)
            action[act_k] = act_v
        return action

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        if raw_action_in is None:
            raw_action_in = brain_info.previous_vector_actions
        # brain_info.previous_vector_actions = raw_action_in
        # total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        # brain_info.action_masks = np.ones((1, total_num_actions))
        return brain_info

    def _revert_actions(self, brain_info:BrainInfo):
        action_in = brain_info.previous_vector_actions
        action = np.zeros(len(self._mlagent_action_space),dtype=int)
        i = 0
        for k,v in self._mlagent_action_space.items():
            VIEW_STEP=9
            if k == 'camera_left_right':
                velocity = action_in['camera'][1]
                rand = np.random.random_sample() * VIEW_STEP
                if velocity > 0 and velocity > rand:
                    action[i]=2
                elif velocity < 0 and velocity < -rand:
                    action[i]=1
            elif k == 'camera_up_down':
                velocity = action_in['camera'][0]
                rand = np.random.random_sample() * VIEW_STEP
                if velocity > 0 and velocity > rand:
                    action[i]=2
                elif velocity < 0 and velocity < -rand:
                    action[i]=1
            elif k in ['craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']:
                action[i]=action_in[k]
            else:
                for move_i, move in enumerate(v):
                    if move in action_in and action_in[move] == 1:
                        action[i]=move_i
            i += 1    
        brain_info.previous_vector_actions = [action]
        # TODO action masks
        # vector_action_space_size = [len(v) for k,v in self._mlagent_action_space.items()]
        return brain_info

    def process_demonstrations(self, brain_info: BrainInfo):
        # revert action
        debug_original_actions = brain_info.previous_vector_actions
        brain_info = self._revert_actions(brain_info)
        # debug check
        # debug_reverted_actions = brain_info.previous_vector_actions
        # debug_reverted_actions = {self.env.spec.id:debug_reverted_actions}
        # debug_check = self._process_action(debug_original_actions)
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info


    def step(self, raw_action_in):
        processed_action = self._process_action(raw_action_in)

        ob, reward, done, info = self.env.step(processed_action)
        max_reached = False
        if 'TimeLimit.truncated' in info:
            max_reached = True
        # if done:
        #     brain_info = self.reset()
        #     brain_info.max_reached = max_reached
        # else:
        brain_info = self._create_brain_info(ob, reward, done, info, raw_action_in[self.env.spec.id], max_reached)
        brain_info = self._process_brain_info(brain_info, raw_action_in[self.env.spec.id])
        return brain_info   

    def _create_brain_info(self, ob, reward = None, done = None, info = None, action = None, max_reached = False)->BrainInfo:
        return MineRLToMLAgentWrapper.create_brain_info(
            ob=ob,
            agent_id=self._agent_id,
            brain_params=self._brain_params,
            reward=reward,
            done = done, 
            info = info, 
            action = action,
            max_reached = max_reached
        )

    @staticmethod
    def get_brain_params(brain_name)->BrainParameters:
        brain_params = MineRLToMLAgentWrapper.static_brain_params[brain_name]
        env = MineRLToMLAgentWrapper.static_wrapped_env[brain_name]
        envs = []
        wrapped_env = env
        while True:
            if hasattr(wrapped_env, 'process_demonstrations'):
                envs.append(wrapped_env)
            wrapped_env = wrapped_env.env
            if not hasattr(wrapped_env, 'env'):
                break 
        brain_params = envs[0].brain_parameters
        return brain_params

    @staticmethod
    def set_wrappers_for_pretraining(brain_name: str, wrapped_env: gym.Wrapper):
        MineRLToMLAgentWrapper.static_wrapped_env[brain_name] = wrapped_env
        MineRLToMLAgentWrapper.static_brain_params[brain_name] = wrapped_env.brain_parameters

    @staticmethod
    def process_brain_info_through_wrapped_envs(brain_name: str, brain_info: BrainInfo):
        env = MineRLToMLAgentWrapper.static_wrapped_env[brain_name]
        envs = []
        wrapped_env = env
        while True:
            if hasattr(wrapped_env, 'process_demonstrations'):
                envs.append(wrapped_env)
            wrapped_env = wrapped_env.env
            if not hasattr(wrapped_env, 'env'):
                break
        for wrapped_env in reversed(envs):
            brain_info = wrapped_env.process_demonstrations(brain_info)
        return brain_info

    @staticmethod
    def create_brain_info(
        ob, agent_id, brain_params, reward = None, done = None, 
        info = None, action = None, max_reached = False
        )->BrainInfo:
        vector_obs = []
        vis_obs = []
        for k,v in ob.items():
            if k == 'pov':
                v = ob['pov']
                v = v.reshape(1,v.shape[0],v.shape[1],v.shape[2])
                vis_obs = v
            elif type(v) is dict or type(v) is OrderedDict:
                for a,b in ob['inventory'].items():
                    vector_obs.append((float)(b))
            else:
                vector_obs.append((float)(v))
        vector_obs = np.array(vector_obs)
        vector_obs = vector_obs.reshape(1, vector_obs.shape[0])
        # vector_obs: List[np.ndarray] = [vector_obs]

        # vis_obs = vis_obs.reshape(1, vis_obs.shape[0], vis_obs.shape[1], vis_obs.shape[2])
        vis_obs = [vis_obs]

        text_obs = []
        memory = np.zeros((0, 0))

        rew = reward if reward is not None else 0.0
        rew = rew if not np.isnan(rew) else 0.0
        rew = [rew]

        local_done = [done] if done is not None else [False]
        text_action = []
        max_reached = [max_reached]
        agents=[agent_id]
        total_num_actions = sum(brain_params.vector_action_space_size)
        mask_actions = np.ones((len(agents), total_num_actions))
        vector_action = action if action is not None else np.zeros((len(agents), len(brain_params.vector_action_space_size)))
        custom_observations = []
        brain_info = BrainInfo(
            visual_observation=vis_obs,
            vector_observation=vector_obs,
            text_observations=text_obs,
            memory=memory,
            reward=rew,
            agents=agents,
            local_done=local_done,
            vector_action=vector_action,
            text_action=text_action,
            max_reached=max_reached,
            action_mask=mask_actions,
            custom_observations=custom_observations
        )
        return brain_info



    def reset(self):
        # self.env._max_episode_steps = 1000 # HACK
        # self.env._max_episode_steps = 2000 # HACK
        ob = self.env.reset()
        brain_info = self._create_brain_info(ob)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_params

    @property
    def vector_obs_keys(self) ->Dict[str, int]:
        return self._vector_obs_keys

    @property
    def agent_id(self) ->str:
        return self._agent_id

    def debug_print(self, env_name):
        minerl_action_space = gym.envs.registry.env_specs[env_name]._kwargs['action_space'].spaces
        minerl_observation_space = gym.envs.registry.env_specs[env_name]._kwargs['observation_space'].spaces
        for k, v in minerl_action_space.items(): 
            print(k, v)
            if type(v) is minerl.env.spaces.Discrete:
                print(' Discrete:', v.n)
                # dtype:dtype('int64')
                # n:2
                # np_random:RandomState(MT19937) at 0x10E16CCA8
                # shape:()            
            elif type(v) is minerl.env.spaces.Box:
                # bounded_above:array([ True,  True])
                # bounded_below:array([ True,  True])
                # dtype:dtype('float32')
                # high:array([180., 180.], dtype=float32)
                # low:array([-180., -180.], dtype=float32)
                # np_random:RandomState(MT19937) at 0x10E166EB8
                # shape:(2,)
                print(' Box shape:', v.shape)
                print(' Box high:', v.high)
                print(' Box low:', v.low)
            elif type(v) is minerl.env.spaces.Enum:
                print(' Enum:', v.n)
                print(' Enum:', v.values)
            else:
                print(' type not handles:', type(v))
        for k, v in minerl_observation_space.items(): 
            print(k, v)
            if type(v) is minerl.env.spaces.Dict:
                print(' Dict:')
                # for s in v.spaces:

            elif type(v) is minerl.env.spaces.Box:
                # bounded_above:array([ True,  True])
                # bounded_below:array([ True,  True])
                # dtype:dtype('float32')
                # high:array([180., 180.], dtype=float32)
                # low:array([-180., -180.], dtype=float32)
                # np_random:RandomState(MT19937) at 0x10E166EB8
                # shape:(2,)
                print(' Box shape:', v.shape)
                print(' Box high:', v.high)
                print(' Box low:', v.low)
            elif type(v) is minerl.env.spaces.Enum:
                print(' Enum:', v.n)
                print(' Enum:', v.values)
            else:
                print(' type not handles:', type(v))        

