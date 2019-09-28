import gym
from mlagents.envs import BrainInfo, BrainParameters
import minerl
import numpy as np


class MineRLToMLAgentWrapper(gym.Wrapper):
    """A class to wrap a MineRL OpenAI.gym environment as a UnityMLAgent environment.

    Specifically, maps observations, actions, rewards, so that we can train with ml-agents
    """

    def __init__(self, env, port):
        gym.Wrapper.__init__(self, env)
        env_id = env.spec.id

        self._minerl_action_space = gym.envs.registry.env_specs[env_id]._kwargs['action_space'].spaces
        self._minerl_observation_space = gym.envs.registry.env_specs[env_id]._kwargs['observation_space'].spaces
        self._mlagent_action_space = dict()
        if 'forward' and 'back' in self._minerl_action_space:
            self._mlagent_action_space['forward_back']=['noop','forward','back']
        if 'left' and 'right' in self._minerl_action_space:
            self._mlagent_action_space['left_right']=['noop','left','right']
        if 'attack' and 'jump' in self._minerl_action_space:
            self._mlagent_action_space['attack_jump']=['noop','attack','jump']
        self._mlagent_action_space['camera_left_right']=['noop','camera_left','camera_right']
        self._mlagent_action_space['camera_up_down']=['noop','camera_up','camera_down']
        if 'sneak' and 'sprint' in self._minerl_action_space:
            self._mlagent_action_space['sneak_sprint']=['noop','sneak','sprint']
        for key in ['craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']:
            if key in self._minerl_action_space: 
                self._mlagent_action_space[key]=[x for x in self._minerl_action_space[key].values]

        brain_name = 'MineRLUnityBrain'
        self._agent_id = brain_name+'-'+str(port)
        vector_observation_space_size = int(0)
        if 'equipped_items' in self._minerl_observation_space:
            for k,v in self._minerl_observation_space['equipped_items'].spaces['mainhand'].spaces.items():
                if type(v) is minerl.env.spaces.Box:
                    vector_observation_space_size += 1
                elif type(v) is minerl.env.spaces.Enum:
                    vector_observation_space_size += len(v)
                else:
                    raise NotImplementedError
        # Box()
        for obs in ['compassAngle']:
            if obs in self._minerl_observation_space:
                vector_observation_space_size += 1
        # Dict()
        for obs in ['inventory']:
            if obs in self._minerl_observation_space:
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

    def step(self, raw_action_in):
        # map mlagent action to minerl
        # raw_action_in['MineRLUnityBrain'][0] = [1,0,2,0,0,0,0]
        raw_action_in = raw_action_in['MineRLUnityBrain']
        action_in = raw_action_in[0]
        action = self.action_space.sample()
        for act_k in action:
        # for mine_k, mine_v in self._minerl_action_space.items():
            act_v = action[act_k]
            if act_k in ['forward', 'back', 'left', 'right', 'sneak', 'sprint', 'jump', 'attack']:
                for i, v in enumerate(self._mlagent_action_space.items()): 
                    if act_k in v[0] and act_k in v[1]:
                        act_v = type(act_v) (action_in[i] == v[1].index(act_k))
            if act_k in ['craft', 'equip', 'nearbyCraft', 'nearbySmelt', 'place']:
                for i, v in enumerate(self._mlagent_action_space.items()): 
                    if act_k in v[0] and act_k in v[1]:
                        act_v = type(act_v) (action_in[i] == v[1].index(act_k))
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
            action[act_k] = act_v

        ob, reward, done, info = self.env.step(action)
        max_reached = False
        if 'TimeLimit.truncated' in info:
            max_reached = True
        # if done:
        #     brain_info = self.reset()
        #     brain_info.max_reached = max_reached
        # else:
        brain_info = self._create_brain_info(ob, reward, done, info, raw_action_in, max_reached)
        return brain_info   

    def _create_brain_info(self, ob, reward = None, done = None, info = None, action = None, max_reached = False)->BrainInfo:
        
        vector_obs = []
        vis_obs = []
        for k,v in ob.items():
            if k == 'pov':
                v = ob['pov']
                v = v.reshape(1,v.shape[0],v.shape[1],v.shape[2])
                vis_obs = v
            elif type(v) is dict:
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
        agents=[self._agent_id]
        total_num_actions = sum(self._brain_params.vector_action_space_size)
        mask_actions = np.ones((len(agents), total_num_actions))
        vector_action = action if action is not None else np.zeros((len(agents), len(self._brain_params.vector_action_space_size)))
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
        self.env._max_episode_steps = 1000 # HACK
        ob = self.env.reset()
        brain_info = self._create_brain_info(ob)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_params

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

