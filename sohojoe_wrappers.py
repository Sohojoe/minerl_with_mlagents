import gym
import time
import numpy as np
from mlagents.envs import AllBrainInfo, BrainInfo, BrainParameters
from collections import deque
import cv2
cv2.ocl.setUseOpenCL(False)

class PruneVisualObservationsWrapper(gym.Wrapper):
    def __init__(self, env, hack_ignor=False):
        super(PruneVisualObservationsWrapper, self).__init__(env)   
        self._parent_brain_parameters = env.brain_parameters
        self._brain_parameters = env.brain_parameters
        self._hack_ignor = hack_ignor
        if self._hack_ignor:
            return

        self._brain_parameters = BrainParameters(
            brain_name = self._parent_brain_parameters.brain_name,
            vector_observation_space_size = self._parent_brain_parameters.vector_observation_space_size, 
            num_stacked_vector_observations = self._parent_brain_parameters.num_stacked_vector_observations,
            camera_resolutions = [],
            vector_action_space_size = self._parent_brain_parameters.vector_action_space_size,
            vector_action_descriptions = self._parent_brain_parameters.vector_action_descriptions,
            vector_action_space_type = 0)   
        self._brain_parameters.number_visual_observations = 0

    def _process_action(self, raw_action_in):
        return raw_action_in

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        # if raw_action_in is None and brain_info is not None:
        #     raw_action_in = brain_info.previous_vector_actions
        if self._hack_ignor:
            return brain_info         
        brain_info.visual_observations = []
        # brain_info.previous_vector_actions = raw_action_in
        # total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        # brain_info.action_masks = np.ones((1, total_num_actions))
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_parameters


class PruneActionsWrapper(gym.Wrapper):
    def __init__(self, env, to_prune):
        super(PruneActionsWrapper, self).__init__(env)   
        self._to_prune = to_prune
        self._parent_brain_parameters = env.brain_parameters

        vector_action_space_size = []
        vector_action_descriptions = []
        for i in range(len(self._parent_brain_parameters.vector_action_space_size)):
            description = self._parent_brain_parameters.vector_action_descriptions[i]
            space_size = self._parent_brain_parameters.vector_action_space_size[i]
            if description not in self._to_prune:
                vector_action_space_size.append(space_size)
                vector_action_descriptions.append(description)

        self._brain_parameters = BrainParameters(
            brain_name = self._parent_brain_parameters.brain_name,
            vector_observation_space_size = self._parent_brain_parameters.vector_observation_space_size, 
            num_stacked_vector_observations = self._parent_brain_parameters.num_stacked_vector_observations,
            camera_resolutions = self._parent_brain_parameters.camera_resolutions,
            vector_action_space_size = vector_action_space_size,
            vector_action_descriptions = vector_action_descriptions,
            vector_action_space_type = 0)

    def _process_action(self, action_in):
        actions = {
            self.brain_parameters.brain_name: self._revert_action(action_in)
        }        
        return actions

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        if raw_action_in is None:
            raw_action_in = brain_info.previous_vector_actions
        brain_info.previous_vector_actions = raw_action_in
        total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        brain_info.action_masks = np.ones((1, total_num_actions))
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        actions = brain_info.previous_vector_actions
        actions = self._convert_action(actions)
        brain_info.previous_vector_actions = actions
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info
    
    def _revert_action(self, action_in):
        actions = []
        action_idx = 0
        for i in range(len(self._parent_brain_parameters.vector_action_space_size)):
            description = self._parent_brain_parameters.vector_action_descriptions[i]
            if description not in self._to_prune:
                action = action_in[0][action_idx]
                actions.append(action)
                action_idx += 1
            else:
                actions.append(0)
        actions = np.array(actions)
        actions = actions.reshape(1, actions.shape[0])
        return actions


    def _convert_action(self, action_in):
        actions = []
        for i in range(len(self._parent_brain_parameters.vector_action_space_size)):
            description = self._parent_brain_parameters.vector_action_descriptions[i]
            action = action_in[0][i]
            if description not in self._to_prune:
                actions.append(action)
        actions = np.array(actions)
        actions = actions.reshape(1, actions.shape[0])
        return actions


    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        actions = self._convert_action(brain_info.previous_vector_actions)
        brain_info.previous_vector_actions = actions
        total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        brain_info.action_masks = np.ones((1, total_num_actions))
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_parameters


num_actions = 0
human_wants_restart = False  
human_agent_action = 0   
human_sets_pause = False
human_has_control = False
human_agent_display = True
up_key = False
pause = False
class KeyboardControlWrapper(gym.Wrapper):
    def __init__(self, env, set_human_control=False, no_diagnal=False):
        super(KeyboardControlWrapper, self).__init__(env)   
        self._env = env
        self._set_human_control = set_human_control
        self._no_diagnal = no_diagnal
        self._last_action = 0

        # env.render()
        # env.unwrapped.viewer.window.on_key_press = key_press
        # env.unwrapped.viewer.window.on_key_release = key_release   
        global human_agent_action, human_wants_restart, human_sets_pause, human_has_control, num_actions, human_agent_display, up_key, pause
        # num_actions = env.action_space.sp.n
        human_wants_restart = False  
        human_agent_action = 0   
        human_sets_pause = False
        human_has_control = self._set_human_control
        human_agent_display = True
        up_key = False
        pause = False
        self._please_lazy_init = True
        self._last_time = time.time()

    def action(self, action):
        global human_agent_action, human_wants_restart, human_sets_pause, human_has_control, num_actions, human_agent_display, up_key, pause
        def key_press(key, mod):
            global human_agent_action, human_wants_restart, human_sets_pause, human_has_control, num_actions, human_agent_display, up_key, pause
            old_action=human_agent_action
            if key==0xff0d: human_wants_restart = True
            if key==65307: human_sets_pause = not human_sets_pause
            if key==65307: human_has_control = not human_has_control
            if key==65362: # up
                if human_agent_action == 3: human_agent_action = 6
                elif human_agent_action == 4: human_agent_action = 7
                else: human_agent_action = 1 
                up_key = True
            if key==65364: human_agent_action = 2 # down
            if key==65361: # left
                if human_agent_action == 1: human_agent_action = 6
                else: human_agent_action = 3
            if key==65363: # right
                if human_agent_action == 1: human_agent_action = 7
                else: human_agent_action = 4
            # if key==113: human_agent_action = 6 # q forward + left
            # if key==119: human_agent_action = 1 # w forward
            # if key==101: human_agent_action = 7 # e forward + right
            # if key==115: human_agent_action = 2 # s back
            # if key==97: human_agent_action = 3 # a back
            # if key==100: human_agent_action = 4 # d back
            if key==112: 
                pause = not pause # p pause
                print ('pause = ', pause)
                return
            if key==32: # space
                human_agent_action = 5 
            if key==65289: human_agent_display = not human_agent_display
            # # 65307 # escape
            # a = int( key - ord('0') ) 
            # if a <= 0 or a >= num_actions: 
            #     if old_action is human_agent_action:
            #         print ('key:', key)
            #     return
            # human_agent_action = a

        def key_release(key, mod):
            global human_agent_action, num_actions, up_key
            # if key==65362: human_agent_action = 0 # up
            if key==65364 and human_agent_action==2: # down
                human_agent_action = 0 
            # if key==65361: human_agent_action = 0 # left
            # if key==65363: human_agent_action = 0 # right
            if key==65362: # up
                if human_agent_action == 6: human_agent_action = 3
                elif human_agent_action == 7: human_agent_action = 4
                elif human_agent_action == 1: human_agent_action=0
                up_key=False
            if key==65361: # left
                if human_agent_action == 6: human_agent_action = 1
                elif human_agent_action == 3: human_agent_action=0
            if key==65363: # right
                if human_agent_action == 7: human_agent_action = 1
                elif human_agent_action == 4: human_agent_action=0

            if key==32: # space
                if up_key:
                    human_agent_action = 1
                else:
                    human_agent_action = 0
            # if key==113: human_agent_action = 0 # q forward + left
            # if key==119: human_agent_action = 0 # w forward
            # if key==101: human_agent_action = 0 # e forward + right
            # if key==115: human_agent_action = 0 # s back
            # if key==97: human_agent_action = 0 # a back
            # if key==100: human_agent_action = 0 # d back
            
            a = int( key - ord('0') )
            if a <= 0 or a >= num_actions: return
            if human_agent_action == a:
                human_agent_action = 0
        if self._please_lazy_init and self.viewer is not None:
            self._please_lazy_init = False
            self.viewer.window.on_key_press = key_press
            self.viewer.window.on_key_release = key_release   
            human_has_control = self._set_human_control
        if human_has_control:
            # while pause:
            #     time.sleep(0.01)
            while time.time()-self._last_time < 1/10.:
                time.sleep(0.01)
            self._last_time = time.time()
            if self._no_diagnal:
                if human_agent_action == 6 or human_agent_action == 7:
                    human_agent_action = self._last_action
            a = human_agent_action
            self._last_action = human_agent_action
            # if human_agent_action is 3 or human_agent_action is 4:
            #     human_agent_action = 0
            return a
        return action


    def _process_action(self, action_in):
        global human_has_control
        key_action = self.action(0)
        if human_has_control:
            action_in[self.env.brain_parameters.brain_name].fill(0)

            i_forward_back = -1
            i_camera_left_right = -1
            i_attack_jump = -1
            try:
                i_camera_left_right = self.brain_parameters.vector_action_descriptions.index('camera_left_right')
                i_forward_back = self.env.brain_parameters.vector_action_descriptions.index('forward_back')
                i_attack_jump = self.env.brain_parameters.vector_action_descriptions.index('attack_jump')
            except AttributeError:
                pass

            if key_action == 1 and i_forward_back > -1: # 1 = up
                action_in[self.brain_parameters.brain_name][0][i_forward_back] = 1
            elif key_action == 2 and i_forward_back > -1: # 2 = down
                action_in[self.brain_parameters.brain_name][0][i_forward_back] = 2
            elif key_action == 3 and i_camera_left_right > -1: # 3 = left
                action_in[self.brain_parameters.brain_name][0][i_camera_left_right] = 1
            elif key_action == 4 and i_camera_left_right > -1: # 4 = right
                action_in[self.brain_parameters.brain_name][0][i_camera_left_right] = 2
            elif key_action == 5: # 5 = space / jump
                action_in[self.brain_parameters.brain_name][0][i_forward_back] = 1 
                action_in[self.brain_parameters.brain_name][0][i_attack_jump] = 2
            elif key_action == 6 and i_forward_back > -1 and i_camera_left_right > -1: # 6 = up left
                action_in[self.brain_parameters.brain_name][0][i_forward_back] = 1
                action_in[self.brain_parameters.brain_name][0][i_camera_left_right] = 1
            elif key_action == 7 and i_forward_back > -1 and i_camera_left_right > -1: # 7 = up right
                action_in[self.brain_parameters.brain_name][0][i_forward_back] = 1
                action_in[self.brain_parameters.brain_name][0][i_camera_left_right] = 2
# 0 'attack_jump':['noop', 'attack', 'jump']
# 1 'camera_left_right':['noop', 'camera_left', 'camera_right']
# 2 'camera_up_down':['noop', 'camera_up', 'camera_down']
# 3 'forward_back':['noop', 'forward', 'back']
# 4 'left_right':['noop', 'left', 'right']
# 5 'place':['none', 'dirt']
# 6 'sneak_sprint':['noop', 'sneak', 'sprint']    
#            
        return action_in

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        if raw_action_in is None:
            raw_action_in = brain_info.previous_vector_actions
        # brain_info.previous_vector_actions = raw_action_in
        # total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        # brain_info.action_masks = np.ones((1, total_num_actions))
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
 
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        self._renderObs(brain_info, True)
        return brain_info

    def reset(self):
        return self.env.reset()

    def _renderObs(self, brain_info, should_render):
        # brain_info.visual_observations, brain_info.vector_observations
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        if not should_render:
            self.viewer.imshow(self._empty)
            return self.viewer.isopen
        # if self._has_vector_obs:
        #     visual_obs = obs['visual'].copy()
        #     vector_obs = obs['vector'].copy()
        # else:
        visual_obs = brain_info.visual_observations[0][0].copy()           
        # if self._has_vector_obs and self._display_vector_obs:
        if True:
            w = 84
            # max_bright = 1
            max_bright = 255
        #     # Displays time left and number of keys on visual observation
            visual_obs[0:10, :, :] = 0
            vector_obs:Dict[str, float] = dict()
            i = 0
            v = brain_info.rewards[0]
            start = int(i * 16.8) + 4
            if v > 0:
                v = min(1.,v)
                v = int(w*v)
                v = max(1,v)
                end = start + v
                visual_obs[6:10, start:end, 1:2] = max_bright
            elif v < 0:
                v = max(-1.,v)
                v = int(w*v)
                v = min(-1,v)
                end = start - v
                visual_obs[6:10, start:end, 0:1] = max_bright
            i = 0
            for k,v in self.vector_obs_keys.items():
                v = brain_info.vector_observations[0][v]
                # if k in 'compassAngle':
                #     v = v/180.
                vector_obs[k]=v
                start = int(i * 16.8) + 4
                if v > 0:
                    v = min(1.,v)
                    end = start + int(10.*v)
                    visual_obs[1:5, start:end, 1:2] = max_bright
                elif v < 0:
                    v = max(-1.,v)
                    end = start - int(10.*v)
                    visual_obs[1:5, start:end, 0:1] = max_bright
                i += 1

        self._8bit = visual_obs
        # if type(visual_obs[0][0][0]) is np.float32 or type(visual_obs[0][0][0]) is np.float64:
            # _8bit = (255.0 * visual_obs).astype(np.uint8)
        # self._8bit = ( visual_obs).astype(np.uint8)
        self.viewer.imshow(self._8bit)
        return self.viewer.isopen

    @property 
    def is_paused(self):
        global pause
        return pause
    def render(self, mode='human'):
        return self._env.render(mode=mode)

class VisualObsAsFloatWrapper(gym.Wrapper):
    def __init__(self, env):
        super(VisualObsAsFloatWrapper, self).__init__(env)   

    def _process_action(self, raw_action_in):
        return raw_action_in

    def _process_brain_info(self, brain_info, raw_action_in=None):
        if len(brain_info.visual_observations) > 0:
            visual_obs = brain_info.visual_observations[0]
            visual_obs = (visual_obs / 255.0).astype(np.float)
            brain_info.visual_observations[0] = visual_obs
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

class NormalizeObservationsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(NormalizeObservationsWrapper, self).__init__(env)   

    def _process_action(self, raw_action_in):
        return raw_action_in

    def _process_brain_info(self, brain_info, raw_action_in=None):
        for k,i in self.vector_obs_keys.items():
            v = brain_info.vector_observations[0][i]
            if k in ['compassAngle']:
                v = v/180.
            # if k in ['inventory']:
            #     v = v/5. # 5 is a lot of an object
            brain_info.vector_observations[0][i] = v
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        action_in = brain_info.previous_vector_actions
        action_out = self._process_action(action_in)
        brain_info.previous_vector_actions = action_out
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

class HardwireActionsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(HardwireActionsWrapper, self).__init__(env)   

    def _process_action(self, raw_action_in):
        action_in = raw_action_in
        if self.brain_parameters.brain_name in raw_action_in:
            action_in = raw_action_in[self.brain_parameters.brain_name]
        actions = []
        action_idx = 0
        for i in range(len(self.brain_parameters.vector_action_space_size)):
            description = self.brain_parameters.vector_action_descriptions[i]
            action = action_in[0][action_idx]
            if description == 'forward_back':
                action = 1
            elif description == 'attack_jump':
                action = 2
            elif description == 'sneak_sprint':
                action = 2
            actions.append(action)
            action_idx += 1
        actions = np.array(actions)
        actions = actions.reshape(1, actions.shape[0])
        if self.brain_parameters.brain_name in raw_action_in:
            actions = {self.brain_parameters.brain_name: actions}
        return actions
        
    def _process_brain_info(self, brain_info, raw_action_in=None):
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        action_in = brain_info.previous_vector_actions
        action_out = self._process_action(action_in)
        brain_info.previous_vector_actions = action_out
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

class RefineObservationsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RefineObservationsWrapper, self).__init__(env)   
        self._parent_brain_parameters = env.brain_parameters

        self._brain_parameters = BrainParameters(
            brain_name = self._parent_brain_parameters.brain_name,
            vector_observation_space_size = self._parent_brain_parameters.vector_observation_space_size, 
            num_stacked_vector_observations = self._parent_brain_parameters.num_stacked_vector_observations,
            camera_resolutions = self._parent_brain_parameters.camera_resolutions,
            vector_action_space_size = self._parent_brain_parameters.vector_action_space_size,
            vector_action_descriptions = self._parent_brain_parameters.vector_action_descriptions,
            vector_action_space_type = 0)
        
        if 'compassAngle' in self.vector_obs_keys:
            i = self.brain_parameters.vector_observation_space_size
            self.vector_obs_keys['north'] = i
            i += 1
            self.vector_obs_keys['east'] = i
            i += 1
            self.vector_obs_keys['south'] = i
            i += 1
            self.vector_obs_keys['west'] = i
            i += 1
            self.brain_parameters.vector_observation_space_size = i

    def _process_action(self, raw_action_in):
        return raw_action_in
        
    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        obs = np.zeros(
            (brain_info.vector_observations.shape[0],self.brain_parameters.vector_observation_space_size), 
            float)
        for k,i in self.vector_obs_keys.items():
            try:
                v = brain_info.vector_observations[0][i]
                if k == 'compassAngle':
                    compassAngle = v
                    north = max(0, abs(180 - (compassAngle) % 360) - 90) / 90.
                    east  = max(0, abs(180 - (compassAngle + 90) % 360) - 90) / 90.
                    south = max(0, abs(180 - (compassAngle + 180) % 360) - 90) / 90.
                    west  = max(0, abs(180 - (compassAngle + 270) % 360) - 90) / 90.
            except IndexError:
                pass
            if k in ['north']:
                v = north
            elif k in ['east']:
                v = east
            elif k in ['south']:
                v = south
            elif k in ['west']:
                v = west
            obs[0][i] = v
        brain_info.vector_observations = obs
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_parameters

class ResetOnDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ResetOnDoneWrapper, self).__init__(env)   
        self._should_reset = False

    def step(self, raw_action_in)->BrainInfo:
        if self._should_reset:
            self.reset()
            # make action = no op
            for action in raw_action_in:
                for i, k in enumerate(action):
                    action[i] = 0

        brain_info = self.env.step(raw_action_in)
        self._should_reset |= brain_info.local_done[0]
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        self._should_reset = False        
        return brain_info

class FrameStackMono(gym.Wrapper):
    def __init__(self, env, n_vis_stack, n_vector_stack):
        """Stack n_vis_stack and n_vector_stack frames.
        """
        gym.Wrapper.__init__(self, env)
        self.n_vis_stack = n_vis_stack
        self.n_vector_stack = n_vector_stack
        self.vector_frames = deque([], maxlen=self.n_vector_stack)
        self.frames = deque([], maxlen=3+(self.n_vis_stack-1))
        self.color_frames = deque([], maxlen=3)
        self.mono_frames = deque([], maxlen=self.n_vis_stack)
        old_pov = env.observation_space.spaces['pov']
        from minerl.env import spaces as minerl_spaces
        pov_shape = (old_pov.shape[0], old_pov.shape[1], 3+(self.n_vis_stack-1))
        pov = minerl_spaces.Box(low=0, high=255, shape=pov_shape, dtype=np.uint8)
        env.observation_space.spaces['pov'] = pov

        self._parent_brain_parameters = env.brain_parameters
        self._brain_parameters = env.brain_parameters
        camera_resolutions = [self._parent_brain_parameters.camera_resolutions[0]]
        mono_camera = {
            "width": camera_resolutions[0]['width'], 
            "height": camera_resolutions[0]['height'], 
            "blackAndWhite": True
            }
        # camera_resolutions = [mono_camera,mono_camera,mono_camera]
        for _ in range(self.n_vis_stack-1):
            camera_resolutions.append(mono_camera)

        self._brain_parameters = BrainParameters(
            brain_name = self._parent_brain_parameters.brain_name,
            vector_observation_space_size = self._parent_brain_parameters.vector_observation_space_size, 
            num_stacked_vector_observations = self.n_vector_stack,
            camera_resolutions = camera_resolutions,
            vector_action_space_size = self._parent_brain_parameters.vector_action_space_size,
            vector_action_descriptions = self._parent_brain_parameters.vector_action_descriptions,
            vector_action_space_type = 0)   

    def _process_action(self, raw_action_in):
        return raw_action_in

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        pov = brain_info.visual_observations[0][0]
        while len(self.mono_frames) < self.mono_frames.maxlen:
            self._add_ob(pov)
        vector_obs = brain_info.vector_observations[0]
        while len(self.vector_frames) < self.vector_frames.maxlen:
            self._add_vector_ob(vector_obs)
        pov = self._get_ob()
        vector_obs = self._get_vector_ob()
        brain_info.visual_observations = pov
        brain_info.vector_observations = vector_obs
        return brain_info

    def process_demonstrations(self, brain_info):
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_parameters

    def reset(self):
        brain_info = self.env.reset()
        pov = brain_info.visual_observations[0][0]
        vector_obs = brain_info.vector_observations[0]
        for _ in range(self.n_vis_stack):
            self._add_ob(pov)
        for _ in range(self.n_vector_stack):
            self._add_vector_ob(vector_obs)
        pov = self._get_ob()
        vector_obs = self._get_vector_ob()
        brain_info.visual_observations = pov
        brain_info.vector_observations = vector_obs
        return brain_info

    def step(self, action):
        brain_info = self.env.step(action)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def _add_ob(self, ob):
        ob_t = ob.T
        self.color_frames.append(ob_t[0])
        self.color_frames.append(ob_t[1])
        self.color_frames.append(ob_t[2])
        mono = cv2.cvtColor(ob.astype(np.float32), cv2.COLOR_RGB2GRAY)
        mono = mono.astype(ob.dtype)
        self.mono_frames.append(mono)

    def _get_ob(self):
        color = [self.color_frames[0]]
        color.append(self.color_frames[1])
        color.append(self.color_frames[2])
        color = np.array(color).T
        color = color.reshape(1, color.shape[0], color.shape[1], color.shape[2])
        cameras = [color]
        for i in range(self.n_vis_stack-1):
            mono = self.mono_frames[i+1]
            mono = mono.reshape(1, mono.shape[0], mono.shape[1], 1)
            cameras.append(mono)
        return cameras

    def _add_vector_ob(self, ob):
        self.vector_frames.append(ob)

    def _get_vector_ob(self):
        vector_obs = []
        for i in range(self.n_vector_stack):
            frame = self.vector_frames[i]
            vector_obs.append(frame)
        vector_obs = np.array(vector_obs)
        vector_obs = vector_obs.reshape(1, vector_obs.shape[0]*vector_obs.shape[1])
        return vector_obs

class DingRewardOnDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DingRewardOnDoneWrapper, self).__init__(env)   
        self._parent_brain_parameters = env.brain_parameters
        self._brain_parameters = env.brain_parameters

    def _process_action(self, raw_action_in):
        return raw_action_in

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        for i in range(len(brain_info.local_done)):
            if brain_info.local_done[i]:
                brain_info.rewards[i] -=1
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_parameters



    # 206,376,37,23,124,75,136,349,6232,1898,185,1020
    # 276,1014,99,3,146,60,218,375,2584,729,120,75
    # 299,371,42,127,126,983,147,105,648,710,83,1101
    # 106,683,170,111,105,108,229,2662,1455,705,198,49
    # 201,713,58,91,149,622,288,125,671,1943,113,4067
    # 54,1072,30,353,65,321,1986,2299,254,776,222,137
    # 81,505,74,215,104,270,165,524,607,1363,207,5197
    # 94,816,59,3,285,58,161,485,893,1161,168,1274
    # 62,944,66,72,183,317,1128,221,3057,784,116,816
    # 251,1155,76,114,162,487,152,1592,185,423,613,3336
    # 506,331,342,145,486,2833,2288,352,1496,1523,169,68
    # 445,277,83,122,87,283,190,1552,139,792,116,320
    # 318,1409,80,494,223,538,931,1105,516,1090,138,2534
    # 422,987,89,151,113,353,156,2158,601,898,193,4332
    # 528,930,53,235,89,546,1209,213,2917,684,127,44
    # 342,433,55,84,105,434,307,392,358,756,388,4740
    # 109,353,44,82,63,316,470,1194,250,670,75,1524
    # 121,695,42,111,53,341,602,144,98,1787,188,2028
    # 98,855,87,3,173,71,428,4490,173,758,179,1694
    # 73,670,79,192,57,272,116,622,217,678,150,3635
    # 282,1742,58,3,156,64,176,1829,1997,878,1129,2423
    # 1656,903,56,101,88,388,1891,804,1858,675,96,64
    # 217,928,97,17,197,118,115,3872,1054,718,111,3446
    # 117,406,49,114,47,238,450,52,369,3474,352,444
    # 209,1051,76,71,216,905,191,3026,228,826,103,4716
    # 740,208,52,110,144,893,250,102,365,404,2086,4266
    # 281,1296,65,3,165,154,172,1289,282,721,107,6449
    # 56,663,25,90,44,604,490,344,629,979,221,1316
    # 333,878,38,63,340,379,1846,79,175,786,69,5167
    # 71,785,112,118,78,215,1755,46,163,1222,3010,401
    # 409,346,176,86,90,378,169,65,439,694,80,1206
    # 89,765,38,104,478,295,294,1844,3101,1546,68,2950
    # 
    # Ave	283	768	78	113	154	435	597	1072	1063	1033	349	2214
    # StdDiv	296	351	58	101	108	493	647	1158	1297	590	612	1866
    # 
    # 579,1119,137,214,262,928,1244,2230,2360,1623,961,4080
class EarlyExitWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EarlyExitWrapper, self).__init__(env)   
        self._parent_brain_parameters = env.brain_parameters
        self._brain_parameters = env.brain_parameters
        self._steps_since_reward = 0
        # self._max_steps = [579,1119,137,214,262,928,1244,2230,2360,1623,961,4080000]
        self._max_steps = [1000,1119,500,500,500,928,1244,2230,2360,1623,961,4080000]

    def _process_action(self, raw_action_in):
        return raw_action_in

    def _process_brain_info(self, brain_info:BrainInfo, raw_action_in=None):
        return brain_info

    def process_demonstrations(self, brain_info):
        # revert action
        # procress brain_info
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        self._steps_since_reward +=1
        if brain_info.rewards[0] != 0:
            self._steps_since_reward_idx += 1
            if self._steps_since_reward_idx >= len(self._max_steps)-1:
                self._steps_since_reward_idx = len(self._max_steps)-1
            self._steps_since_reward = 0
        else:
            if self._steps_since_reward > self._max_steps[self._steps_since_reward_idx]:
                brain_info.local_done = [True for i in brain_info.local_done]

        return brain_info

    def reset(self, **kwargs):
        self._steps_since_reward = 0
        self._steps_since_reward_idx = 0
        brain_info =  self.env.reset(**kwargs)
        brain_info = self._process_brain_info(brain_info)
        return brain_info

    @property
    def brain_parameters(self) ->BrainParameters:
        return self._brain_parameters
