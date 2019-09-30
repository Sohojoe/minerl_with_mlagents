import gym
import time
import numpy as np
from mlagents.envs import AllBrainInfo, BrainInfo, BrainParameters

class PruneVisualObservationsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(PruneVisualObservationsWrapper, self).__init__(env)   
        self._parent_brain_parameters = env.brain_parameters

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

    def _process_brain_info(self, brain_info, raw_action_in):
        brain_info.visual_observations = []
        # brain_info.previous_vector_actions = raw_action_in
        # total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        # brain_info.action_masks = np.ones((1, total_num_actions))
        return brain_info

    def step(self, raw_action_in):
        action_in = self._process_action(raw_action_in)
        brain_info = self.env.step(action_in)
        brain_info = self._process_brain_info(brain_info, raw_action_in)
        return brain_info

    def reset(self, **kwargs):
        brain_info =  self.env.reset(**kwargs)
        brain_info.visual_observations = []
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
        # actions = {
        #     self.brain_parameters.brain_name: self._revert_action(action_in[self.brain_parameters.brain_name])
        # }
        actions = {
            self.brain_parameters.brain_name: self._revert_action(action_in)
        }        
        return actions

    def _process_brain_info(self, brain_info, raw_action_in):
        brain_info.previous_vector_actions = raw_action_in
        total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        brain_info.action_masks = np.ones((1, total_num_actions))
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

            i_forward_back = list(self.env._mlagent_action_space.keys()).index('forward_back')
            i_camera_left_right = list(self.env._mlagent_action_space.keys()).index('camera_left_right')
            i_attack_jump = list(self.env._mlagent_action_space.keys()).index('attack_jump')

            if key_action == 1: # 1 = up
                action_in[self.env.brain_parameters.brain_name][0][i_forward_back] = 1
            elif key_action == 2: # 2 = down
                action_in[self.env.brain_parameters.brain_name][0][i_forward_back] = 2
            elif key_action == 3: # 3 = left
                action_in[self.env.brain_parameters.brain_name][0][i_camera_left_right] = 1
            elif key_action == 4: # 4 = right
                action_in[self.env.brain_parameters.brain_name][0][i_camera_left_right] = 2
            elif key_action == 5: # 5 = space / jump
                action_in[self.env.brain_parameters.brain_name][0][i_forward_back] = 1 
                action_in[self.env.brain_parameters.brain_name][0][i_attack_jump] = 2
            elif key_action == 6: # 6 = up left
                action_in[self.env.brain_parameters.brain_name][0][i_forward_back] = 1
                action_in[self.env.brain_parameters.brain_name][0][i_camera_left_right] = 1
            elif key_action == 7: # 7 = up right
                action_in[self.env.brain_parameters.brain_name][0][i_forward_back] = 1
                action_in[self.env.brain_parameters.brain_name][0][i_camera_left_right] = 2
# 0 'attack_jump':['noop', 'attack', 'jump']
# 1 'camera_left_right':['noop', 'camera_left', 'camera_right']
# 2 'camera_up_down':['noop', 'camera_up', 'camera_down']
# 3 'forward_back':['noop', 'forward', 'back']
# 4 'left_right':['noop', 'left', 'right']
# 5 'place':['none', 'dirt']
# 6 'sneak_sprint':['noop', 'sneak', 'sprint']    
#            
        return action_in

    def _process_brain_info(self, brain_info, raw_action_in):
        # brain_info.previous_vector_actions = raw_action_in
        # total_num_actions = sum(self.brain_parameters.vector_action_space_size)
        # brain_info.action_masks = np.ones((1, total_num_actions))
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
                visual_obs[6:10, start:end, 0:1] = max_bright
            elif v < 0:
                v = max(-1.,v)
                v = int(w*v)
                v = min(-1,v)
                end = start - v
                visual_obs[6:10, start:end, 1:2] = max_bright
            i = 0
            for k,v in self.vector_obs_keys.items():
                v = brain_info.vector_observations[0][v]
                if k in 'compassAngle':
                    v = v/180.
                vector_obs[k]=v
                start = int(i * 16.8) + 4
                if v > 0:
                    v = min(1.,v)
                    end = start + int(10.*v)
                    visual_obs[1:5, start:end, 0:1] = max_bright
                elif v < 0:
                    v = max(-1.,v)
                    end = start - int(10.*v)
                    visual_obs[1:5, start:end, 1:2] = max_bright
                i += 1

            key = list(vector_obs.values())[0]
            time_num = list(vector_obs.values())[1]
            # key_num = np.argmax(key, axis=0)
        #     for i in range(key_num):
        #         start = int(i * 16.8) + 4
        #         end = start + 10
        #         visual_obs[1:5, start:end, 0:2] = max_bright
        #     visual_obs[6:10, 0:int(time_num * w), 1] = max_bright    
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