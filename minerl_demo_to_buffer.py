import pathlib
import logging
import os
from typing import List, Tuple
from mlagents.trainers.buffer import Buffer
from mlagents.envs.brain import BrainParameters, BrainInfo
# from mlagents.envs.communicator_objects import (
#     # AgentInfoProto,
#     # BrainParametersProto,
#     # DemonstrationMetaProto,
# )
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore
import minerl
from mlagents.trainers.demo_loader import make_demo_buffer
import random
import numpy as np
from minerl_to_mlagent_wrapper import MineRLToMLAgentWrapper
class Object(object):
    pass

logger = logging.getLogger("mlagents.trainers")

def demo_to_buffer(
    file_path: str, sequence_length: int
) -> Tuple[BrainParameters, Buffer]:
    """
    Loads demonstration file and uses it to fill training buffer.
    :param file_path: Location of demonstration file (.demo).
    :param sequence_length: Length of trajectories to fill buffer.
    :return:
    """

    # early exit if inference mode
    # export EVALUATION_STAGE='testing'
    EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', '')
    if EVALUATION_STAGE == 'testing':
        demo_buffer = Buffer()
        brain_params = MineRLToMLAgentWrapper.get_brain_params(file_path)
        return brain_params, demo_buffer


    # # The dataset is available in data/ directory from repository root.
    # MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

    logger.info("Building data pipeline for {}".format(file_path))
    data = minerl.data.make(file_path)

    report_trajs = []

    trajs = [
        'v1_other_pomegranite_orc-12_24007-29518', 'v1_right_mushroom_fire-breathing_dragon_41653-47509', 'v1_juvenile_apple_angel-7_205561-212353', 
        'v1_juvenile_apple_angel-6_221-11831', 'v1_equal_olive_chimera-7_10379-19453', 'v1_unselfish_blood_orange_savage-18_19656-23843', 
        'v1_other_pomegranite_orc-12_31579-36826', 'v1_svelte_cherry_devil-17_314-11959', 'v1_agonizing_kale_tree_nymph-7_133235-141843', 
        'v1_unselfish_blood_orange_savage-18_14639-19416', 'v1_unselfish_blood_orange_savage-18_399-10066', 'v1_courageous_rutabaga_nessie-1_3069-13764', 
        'v1_agonizing_kale_tree_nymph-20_289-7919', 'v1_right_mushroom_fire-breathing_dragon_88565-95177', 'v1_last_prune_swamp_monster-2_2208-8442', 
        'v1_excellent_mango_beast-6_43472-48953', 'v1_bogus_guava_djinn-17_23146-31716', 'v1_splendid_brussels_sprout_pegasus-5_45696-54118', 
        'v1_agonizing_kale_tree_nymph-7_106750-114380', 'v1_right_mushroom_fire-breathing_dragon_7211-17977', 'v1_agonizing_kale_tree_nymph-20_7989-16044', 
        'v1_excellent_mango_beast-6_20909-29943', 'v1_villainous_black_eyed_peas_loch_ness_monster-1_82621-93105', 'v1_subtle_iceberg_lettuce_nymph-4_16111-20545', 
        'v1_agonizing_kale_tree_nymph-7_74962-82761', 'v1_juvenile_apple_angel-5_4254-15273', 'v1_conscious_tangerine_rain_bird-23_48769-59333', 
        'v1_absolute_grape_changeling-6_37339-46767', 'v1_equal_olive_chimera-9_14563-24740', 'v1_juvenile_apple_angel-7_158092-167444', 
        'v1_bogus_guava_djinn-2_19159-30071', 'v1_other_pomegranite_orc-12_16800-22992']
    # trajs = data.get_trajectory_names()

    all_demo = dict()
    brain_infos = []
    brain_params = MineRLToMLAgentWrapper.get_brain_params(file_path)
    agent_id = 'fake_id'
    # stream_name = random.choice(trajs)    
    for stream_name in trajs:
        demo = Object()
        logger.info("Loading data for {}...".format(stream_name))
        demo.data_frames = list(data.load_data(stream_name, include_metadata=True))
        demo.meta = demo.data_frames[0][-1] 
        cum_rewards = np.cumsum([x[2] for x in demo.data_frames])
        demo.file_len = len(demo.data_frames)
        logger.info("Data loading complete!".format(stream_name))
        logger.info("META DATA: {}".format(demo.meta))
        demo.height, demo.width = data.observation_space.spaces['pov'].shape[:2]
        # all_demo[stream_name]=demo

        if not demo.meta['success']:
            logger.info("SKIP as success=False")
            continue
        if int(demo.meta['duration_steps']) > 12000:
            logger.info("****HACK**** SKIP as > 12k steps")
            continue
        if int(demo.meta['total_reward']) < 1024:
            logger.info("ERROR score must be > 1024 because of dimond = 1024 points")
            continue

        logger.info("*** PASSED CHECKS ****")
        report_trajs.append(stream_name)

        running_reward=0
        for i, frame in enumerate(demo.data_frames):
            ob=frame[0]
            action=frame[1]
            # action=np.hstack([v for v in action.values()])
            reward=float(frame[2])
            ob=frame[3]
            done=frame[4]
            meta_data=frame[5]
            running_reward+=reward
            info={
                'stream_name': meta_data['stream_name'],
                'duration_steps': meta_data['duration_steps'],
                'total_reward': meta_data['total_reward'],
                'success': meta_data['success'],
                'step': i,
                'running_reward':running_reward
                }
            max_reached= i+1 == meta_data['duration_steps']
            brain_info = MineRLToMLAgentWrapper.create_brain_info(
                ob=ob, 
                agent_id=agent_id, brain_params=brain_params, 
                reward = reward, done = done, 
                info = info, action = action, max_reached = max_reached)
            brain_info = MineRLToMLAgentWrapper.process_brain_info_through_wrapped_envs(
                file_path, brain_info)
            brain_infos.append(brain_info)

            del frame[3] # obs, free for memory
        del demo.data_frames
        del demo
        import gc
        gc.collect()

    # brain_params, brain_infos, _ = load_demonstration(file_path)
    demo_buffer = make_demo_buffer(brain_infos, brain_params, sequence_length)

    del brain_infos
    import gc
    gc.collect()

    logger.info("report_trajs = " + str([str(i) for i in report_trajs]))

    return brain_params, demo_buffer
