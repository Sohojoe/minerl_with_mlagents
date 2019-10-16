import minerl
import gym
import logging
import os


def main():
    MINERL_DATA_ROOT = os.getenv(
        'MINERL_DATA_ROOT', '/Development/Analog/mine_rl_submission/data')
    os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT
    # do your main minerl code
    # logging.basicConfig(level=logging.DEBUG)
    data = minerl.data.make("MineRLObtainDiamondDense-v0")
    # success = [
    #     'v1_absolute_grape_changeling-17_418-2303', 'v1_remorseful_current_savage-6_15132-33033', 'v1_villainous_black_eyed_peas_loch_ness_monster-1_62828-69969', 'v1_long_term_okra_dwarf-1_130-3077', 'v1_right_mushroom_fire-breathing_dragon_41653-47509', 'v1_courageous_rutabaga_nessie-1_3069-13764', 'v1_absolute_grape_changeling-6_37339-46767', 'v1_long_term_okra_dwarf-1_44519-79136', 'v1_right_mushroom_fire-breathing_dragon_7211-17977', 'v1_villainous_black_eyed_peas_loch_ness_monster-1_82621-93105', 'v1_subtle_iceberg_lettuce_nymph-2_38140-56582', 'v1_remorseful_current_savage-6_594-14990', 'v1_juvenile_apple_angel-6_221-11831', 'v1_villainous_black_eyed_peas_loch_ness_monster-1_70054-73712', 'v1_agonizing_kale_tree_nymph-7_74962-82761', 'v1_long_term_okra_dwarf-1_3120-16344', 'v1_cute_breadfruit_spirit-1_40823-68203', 'v1_villainous_black_eyed_peas_loch_ness_monster-2_3997-48203', 'v1_villainous_black_eyed_peas_loch_ness_monster-1_26657-34903', 'v1_agonizing_kale_tree_nymph-7_133235-141843', 'v1_long_term_okra_dwarf-1_16451-43284', 'v1_flustered_tuber_doppelganger-1_97912-144591', 'v1_juvenile_apple_angel-5_4254-15273', 'v1_agonizing_kale_tree_nymph-7_106750-114380', 'v1_cute_breadfruit_spirit-1_5117-20930', 'v1_bogus_guava_djinn-2_19159-30071', 'v1_right_mushroom_fire-breathing_dragon_88565-95177', 'v1_lost_chard_changeling-2_450-15377', 'v1_cute_breadfruit_spirit-1_35476-37415', 'v1_right_mushroom_fire-breathing_dragon_47892-72622'
    # ]

    # fail = [
    #     'v1_key_nectarine_spirit-1_1619-3682', 'v1_bogus_guava_djinn-17_23146-31716', 'v1_kindly_lemon_mummy-11_6412-15431', 'v1_juvenile_apple_angel-7_48978-62020', 'v1_aggravating_artichoke_harpy-11_12179-28686', 'v1_equal_olive_chimera-7_10379-19453', 'v1_self_reliant_fig_doppelganger-1_37451-107047', 'v1_unselfish_blood_orange_savage-18_399-10066', 'v1_kindly_lemon_mummy-2_35249-54498', 'v1_agonizing_kale_tree_nymph-20_58203-59745', 'v1_subtle_iceberg_lettuce_nymph-4_32796-57962', 'v1_familiar_endive_ghost-2_9213-38067', 'v1_excellent_mango_beast-6_30058-43288', 'v1_juvenile_apple_angel-26_862-2906', 'v1_bogus_guava_djinn-17_6593-22816', 'v1_juvenile_apple_angel-7_268-28190', 'v1_conscious_tangerine_rain_bird-23_48769-59333', 'v1_ample_salad_doppelganger-1_556-12734', 'v1_velvety_sprouts_tree_nymph-5_10447-17008', 'v1_equal_olive_chimera-7_29496-36896', 'v1_other_pomegranite_orc-12_16800-22992', 'v1_glistening_okra_golum-9_153840-172647', 'v1_juvenile_apple_angel-7_117738-131163', 'v1_red_guava_merman-2_23281-60729', 'v1_quiet_mandarin_orange_ghoul-13_257-145846', 'v1_agonizing_kale_tree_nymph-20_16180-39256', 'v1_velvety_sprouts_tree_nymph-5_193-10356', 'v1_excellent_mango_beast-11_384-31798', 'v1_unselfish_blood_orange_savage-14_21998-38655', 'v1_earnest_water_chestnut_gargoyle-6_255-25556', 'v1_aggravating_artichoke_harpy-11_186-12079', 'v1_cheery_acorn_squash_spirit-1_37168-47677', 'v1_agonizing_kale_tree_nymph-4_41105-57631', 'v1_unselfish_blood_orange_savage-18_14639-19416', 'v1_subtle_iceberg_lettuce_nymph-4_20999-27579', 'v1_equal_olive_chimera-9_25021-26077', 'v1_svelte_cherry_devil-17_314-11959', 'v1_juvenile_apple_angel-7_158092-167444', 'v1_kindly_lemon_mummy-11_17437-48154', 'v1_other_pomegranite_orc-12_24007-29518', 'v1_ample_salad_doppelganger-1_14665-50183', 'v1_all_orange_djinn-4_193-66220', 'v1_kindly_lemon_mummy-2_3088-28614', 'v1_inferior_parsnip_banshee-2_21612-28374', 'v1_wary_salsa_werewolf-4_196-20369', 'v1_excellent_mango_beast-6_49144-52389', 'v1_alarming_arugula_medusa-12_32515-56508', 'v1_conscious_tangerine_rain_bird-23_316-12797', 'v1_juvenile_apple_angel-7_167478-182832', 'v1_agonizing_kale_tree_nymph-20_7989-16044', 'v1_cool_sweet_potato_nymph-13_6646-8392', 'v1_equal_olive_chimera-9_14563-24740', 'v1_excellent_mango_beast-6_20909-29943', 'v1_anxious_lemon_lake_spirit-1_368-24956', 'v1_juvenile_apple_angel-7_205561-212353', 'v1_alarming_arugula_medusa-12_962-31988', 'v1_kindly_lemon_mummy-2_785-3011', 'v1_frozen_date_godzilla-7_10759-37132', 'v1_glistening_okra_golum-9_129195-148178', 'v1_agonizing_kale_tree_nymph-4_4131-16401', 'v1_agonizing_kale_tree_nymph-20_289-7919', 'v1_wary_salsa_werewolf-4_20915-42385', 'v1_equal_olive_chimera-9_221-12864', 'v1_ugly_cress_orc-1_23174-49353', 'v1_tangible_kumquat_giant-11_494-47621', 'v1_agonizing_kale_tree_nymph-20_43999-57348', 'v1_excellent_mango_beast-6_43472-48953', 'v1_subtle_iceberg_lettuce_nymph-12_29391-43991', 'v1_unselfish_blood_orange_savage-15_17061-30979', 'v1_unselfish_blood_orange_savage-18_19656-23843', 'v1_juicy_jackfruit_griffin-1_1316-43029', 'v1_last_prune_swamp_monster-2_2208-8442', 'v1_juvenile_apple_angel-7_78901-93820', 'v1_bogus_guava_djinn-17_35972-57260', 'v1_other_pomegranite_orc-12_31579-36826', 'v1_other_pomegranite_orc-12_351-13496', 'v1_splendid_brussels_sprout_pegasus-5_45696-54118', 'v1_agonizing_kale_tree_nymph-4_580-4067', 'v1_masculine_avocado_ogre-4_37863-64346', 'v1_absolute_grape_changeling-9_4643-34456', 'v1_splendid_brussels_sprout_pegasus-5_429-8799', 'v1_subtle_iceberg_lettuce_nymph-4_16111-20545', 'v1_sticky_chick_pea_gnome-18_17559-31315', 'v1_absolute_zucchini_basilisk-14_12791-21691', 'v1_kindly_lemon_mummy-11_2752-6180', 'v1_agonizing_kale_tree_nymph-20_39345-43897', 'v1_conscious_tangerine_rain_bird-23_12946-47287', 'v1_late_basil_lake_spirit-7_1236-19762'
    # ]
    success = [
        'v1_subtle_iceberg_lettuce_nymph-2_38140-56582', 'v1_sticky_chick_pea_gnome-18_17559-31315', 'v1_ample_salad_doppelganger-1_556-12734', 'v1_equal_olive_chimera-9_14563-24740', 'v1_agonizing_kale_tree_nymph-20_289-7919', 'v1_other_pomegranite_orc-12_351-13496', 'v1_long_term_okra_dwarf-1_16451-43284', 'v1_agonizing_kale_tree_nymph-7_133235-141843', 'v1_subtle_iceberg_lettuce_nymph-4_16111-20545', 'v1_long_term_okra_dwarf-1_16451-43284', 'v1_excellent_mango_beast-6_43472-48953', 'v1_agonizing_kale_tree_nymph-20_16180-39256', 'v1_juvenile_apple_angel-7_117738-131163', 'v1_right_mushroom_fire-breathing_dragon_7211-17977', 'v1_ugly_cress_orc-1_23174-49353', 'v1_familiar_endive_ghost-2_9213-38067', 'v1_glistening_okra_golum-9_129195-148178', 'v1_subtle_iceberg_lettuce_nymph-12_29391-43991', 'v1_agonizing_kale_tree_nymph-7_106750-114380', 'v1_aggravating_artichoke_harpy-11_12179-28686', 'v1_all_orange_djinn-4_193-66220', 'v1_villainous_black_eyed_peas_loch_ness_monster-1_82621-93105', 'v1_kindly_lemon_mummy-2_35249-54498', 'v1_svelte_cherry_devil-17_314-11959', 'v1_all_orange_djinn-4_193-66220', 'v1_courageous_rutabaga_nessie-1_3069-13764', 'v1_cute_breadfruit_spirit-1_5117-20930', 'v1_other_pomegranite_orc-12_24007-29518', 'v1_excellent_mango_beast-6_20909-29943', 'v1_bogus_guava_djinn-17_6593-22816', 'v1_remorseful_current_savage-6_594-14990', 'v1_lost_chard_changeling-2_450-15377', 'v1_juvenile_apple_angel-7_205561-212353', 'v1_juicy_jackfruit_griffin-1_1316-43029', 'v1_remorseful_current_savage-6_15132-33033', 'v1_unselfish_blood_orange_savage-18_14639-19416', 'v1_juvenile_apple_angel-7_158092-167444', 'v1_unselfish_blood_orange_savage-18_399-10066', 'v1_bogus_guava_djinn-17_23146-31716', 'v1_conscious_tangerine_rain_bird-23_12946-47287', 'v1_equal_olive_chimera-7_10379-19453', 'v1_kindly_lemon_mummy-2_3088-28614', 'v1_excellent_mango_beast-11_384-31798', 'v1_excellent_mango_beast-11_384-31798', 'v1_conscious_tangerine_rain_bird-23_48769-59333', 'v1_bogus_guava_djinn-17_35972-57260', 'v1_other_pomegranite_orc-12_16800-22992', 'v1_splendid_brussels_sprout_pegasus-5_45696-54118', 'v1_masculine_avocado_ogre-4_37863-64346', 'v1_other_pomegranite_orc-12_31579-36826', 'v1_masculine_avocado_ogre-4_37863-64346', 'v1_right_mushroom_fire-breathing_dragon_41653-47509', 'v1_agonizing_kale_tree_nymph-4_4131-16401', 'v1_kindly_lemon_mummy-11_17437-48154', 'v1_right_mushroom_fire-breathing_dragon_88565-95177', 'v1_juvenile_apple_angel-5_4254-15273', 'v1_tangible_kumquat_giant-11_494-47621', 'v1_last_prune_swamp_monster-2_2208-8442', 'v1_alarming_arugula_medusa-12_32515-56508', 'v1_tangible_kumquat_giant-11_494-47621', 'v1_unselfish_blood_orange_savage-15_17061-30979', 'v1_flustered_tuber_doppelganger-1_97912-144591', 'v1_long_term_okra_dwarf-1_44519-79136', 'v1_flustered_tuber_doppelganger-1_97912-144591', 'v1_juvenile_apple_angel-6_221-11831', 'v1_absolute_grape_changeling-6_37339-46767', 'v1_bogus_guava_djinn-2_19159-30071', 'v1_agonizing_kale_tree_nymph-7_74962-82761', 'v1_cute_breadfruit_spirit-1_40823-68203', 'v1_frozen_date_godzilla-7_10759-37132', 'v1_earnest_water_chestnut_gargoyle-6_255-25556', 'v1_agonizing_kale_tree_nymph-20_7989-16044', 'v1_frozen_date_godzilla-7_10759-37132', 'v1_unselfish_blood_orange_savage-18_19656-23843', 'v1_excellent_mango_beast-6_30058-43288'
    ]

    # success = []
    # fail = []
    # for _, _, _, _, _, metadata \
    #         in data.sarsd_iter(num_epochs=1, max_sequence_len=32, include_metadata=True):
    #     stream_name = metadata['stream_name']
    #     if (metadata['stream_name'] not in success and metadata['stream_name'] not in fail):
    #         if (metadata['success']):
    #             success.append(stream_name)
    #         else:
    #             fail.append(stream_name)
    # success = []
    # for _, _, rewards, _, _, metadata \
    #             in data.sarsd_iter(num_epochs=1, max_sequence_len=128, include_metadata=True):
    #         stream_name = metadata['stream_name']
    #         if max(rewards) == 1024:
    #             success.append(stream_name)

    env = gym.make('MineRLObtainDiamondDense-v0')
    # env.reset()
    done = False
    episode = 0
    step = 0

    # Sample some data from the dataset!
    # Iterate through a single epoch using sequences of at most 32 steps
    for stream_name in success:
        for state, rec_action, reward, next_state, done, metadata \
                in data.load_data(stream_name, include_metadata=True):
            # for current_states, actions, rewards, next_states, dones, metadata \
            # in data.sarsd_iter(num_epochs=1, max_sequence_len=32, include_metadata=True):
            # print (metadata)
            if (step is 0):
                print("episode", episode)
                print(metadata)
                # env.seed(56582)
                env.reset()
                found_dimond = False
            action = env.action_space.sample()
            for k, v in action.items():
                # for key in action.keys():
                # action[key] = rec_action[key]
                if type(v) is str or type(v) is int:
                    action[k] = type(v)(rec_action[k])
                else:
                    action[k] = rec_action[k]
            # print (action)
            e_obs, e_rew, e_done, _ = env.step(action)
            if (e_rew > 0 or reward > 0):
                print("step:", step, e_rew, reward)
            if (e_rew == 1024 or reward == 1024):
                found_dimond = True
            step += 1
            if done or e_done:
                env.reset()
            if (step >= metadata['duration_steps']):
                step = 0
                episode += 1

    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    main()
