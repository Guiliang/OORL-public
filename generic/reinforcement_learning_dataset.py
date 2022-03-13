import os
import glob
import random

import gym
import textworld.gym


def get_training_game_env(data_dir,
                          difficulty_level,
                          training_size,
                          requested_infos=None,
                          max_episode_steps=None,
                          batch_size=None,
                          if_partial_game=False):
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 99, 'mixed']
    assert training_size in [1, 20, 100]

    if difficulty_level == 'mixed':
        difficulty_levels = [3, 5, 7, 9]
        max_level_game = 25
    else:
        difficulty_levels = [difficulty_level]
        max_level_game = 100

    # training games
    game_file_names = []
    for difficulty_level in difficulty_levels:
        game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
        if os.path.isdir(game_path):
            game_file_names += glob.glob(os.path.join(game_path, "*.z8"))[: max_level_game]
        else:
            game_file_names.append(game_path)
    if if_partial_game:
        game_file_names = game_file_names[:batch_size]
    if requested_infos is not None:
        env_id = textworld.gym.register_games(sorted(game_file_names), request_infos=requested_infos,
                                              max_episode_steps=max_episode_steps, batch_size=batch_size,
                                              name="training", asynchronous=False, auto_reset=False)
        env = gym.make(env_id)
        num_game = len(game_file_names)
        return env, num_game
    else:
        return game_file_names


def get_evaluation_game_env(data_dir, difficulty_level, valid_or_test="valid",
                            requested_infos=None,
                            max_episode_steps=None,
                            batch_size=None,
                            debug_mode=False
                            ):
    assert valid_or_test in ["valid", "test"]
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 99, 'mixed']

    if difficulty_level == 'mixed':
        difficulty_levels = [3, 5, 7, 9]
        max_level_game = 5 if not debug_mode else 20
    else:
        difficulty_levels = [difficulty_level]
        max_level_game = 20

    # eval games
    game_file_names = []
    for difficulty_level in difficulty_levels:
        game_path = data_dir + "/" + valid_or_test + "/difficulty_level_" + str(difficulty_level)
        if os.path.isdir(game_path):
            game_file_names += glob.glob(os.path.join(game_path, "*.z8"))[: max_level_game]
        else:
            game_file_names.append(game_path)

    if requested_infos is not None:
        env_id = textworld.gym.register_games(sorted(game_file_names), request_infos=requested_infos,
                                              max_episode_steps=max_episode_steps, batch_size=batch_size,
                                              name="eval", asynchronous=False, auto_reset=False)
        env = gym.make(env_id)
        num_game = len(game_file_names)
        return env, num_game
    else:
        return game_file_names
