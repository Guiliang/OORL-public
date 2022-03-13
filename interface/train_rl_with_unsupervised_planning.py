import ast
import datetime
import json
import os
import random
import copy
import numpy as np
import torch
import gc

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from evaluate.evaluate_rl import evaluate_rl_with_unsupervised_graphs
from agent.agent import OORLAgent
from generic import reinforcement_learning_dataset
from generic.data_utils import load_config, read_args, get_goal_sentence, generate_triplets_filter_mask, \
    adj_to_triplets, matching_object_from_obs, handle_ingame_rewards, process_facts, serialize_facts, \
    extract_goal_sentence_from_obs, rl_force_explore, get_game_difficulty_level
from generic.model_utils import HistoryScoreCache, load_graph_extractor, to_pt, to_np, memory_usage_psutil
from planner.mcts_planner import MCTSPlanning


def test(args):
    raise EnvironmentError("Please use the 'test_rl_planning.py' for testing.")


def train(args):
    random.seed(args.SEED)
    time_1 = datetime.datetime.now()
    today = datetime.date.today()
    manual_explore = True
    # min_unexplore_rate = 0.1
    config, debug_mode, log_file_path = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    print("Apply{0} manual explore.".format('' if manual_explore else ' no'), file=log_file, flush=True)
    print("Start Training.", file=log_file, flush=True)
    debug_msg = ''  # for local machine debugging
    if debug_mode:
        debug_msg = '_debug'
        config['rl']['epsilon_greedy']['epsilon_anneal_from'] = 0
        config['general']['training']['batch_size'] = 8
        config['rl']['training']['max_nb_steps_per_episode'] = 200
        config['rl']['evaluate']['max_nb_steps_per_episode'] = 20
        config['rl']['training']['target_net_update_frequency'] = 50
        config['rl']['training']['learn_start_from_this_episode'] = 1
        config['rl']['replay']['update_per_k_game_steps'] = 10
        config['rl']['replay']['replay_batch_size'] = 16
        config['rl']['replay']['replay_memory_capacity'] = 100
        config['rl']['epsilon_greedy']['epsilon_anneal_episodes'] = 50
        if_partial_game = False
    else:
        if_partial_game = False

    # if_partial_game = True
    # config['general']['training']['batch_size'] = 2
    # config['rl']['epsilon_greedy']['epsilon_anneal_episodes'] = 500
    # print('-' * 100, file=log_file, flush=True)
    # print("*** Warning: Launching a toy experiment with few games. ***", file=log_file, flush=True)
    # print('-' * 100, file=log_file, flush=True)
    step_in_total = 0
    episode_no = 0
    perfect_training = 0
    best_train_performance_so_far, best_eval_performance_so_far = 0.0, 0.0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_game_points_normalized = HistoryScoreCache(capacity=500)
    running_avg_graph_rewards = HistoryScoreCache(capacity=500)
    running_avg_count_rewards = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)
    running_avg_dqn_loss = HistoryScoreCache(capacity=500)
    running_avg_game_rewards = HistoryScoreCache(capacity=500)

    print("\n\n" + "*" * 30 + "Start Loading Pretained Model" + "*" * 30, file=log_file, flush=True)
    agent = OORLAgent(config, log_file=log_file, debug_msg=debug_msg, seed=args.SEED)
    if args.FIX_POINT is None:
        save_date_str = today.strftime("%b-%d-%Y")
    else:
        save_date_str = args.FIX_POINT
    save_to_path = agent.output_dir + agent.experiment_tag + "/difficulty_level_{2}/saved_model_dqn_unsupervised_df-" \
                                                             "{2}-mem-{3}-epi-{4}-maxstep-{5}-anneal-{6}-{7}{8}{9}{10}_{0}{1}.pt".format(
        save_date_str,
        debug_msg,
        agent.difficulty_level,
        agent.dqn_memory.capacity,
        agent.epsilon_anneal_episodes,
        agent.max_nb_steps_per_episode,
        agent.epsilon_anneal_to,
        'cstr' if agent.apply_goal_constraint else '',
        '-scratch' if not manual_explore else '',
        '-me-{0}'.format(agent.min_unexplore_rate) if agent.min_unexplore_rate is not None else '',
        '-seed-{0}'.format(args.SEED)
    )
    if args.FIX_POINT is not None:
        load_keys, episode_no, loss, acc, running_game_points = agent.load_pretrained_model(load_from=save_to_path,
                                                                                            log_file=log_file,
                                                                                            load_running_records=True)
        agent.update_target_net()
        agent.epsilon = agent.epsilon_scheduler.value(episode_no - agent.learn_start_from_this_episode)
        agent.epsilon = max(agent.epsilon, 0.0)
        agent.unexplore_rate = agent.unexplore_scheduler.value(episode_no - agent.learn_start_from_this_episode)
        agent.unexplore_rate = max(agent.unexplore_rate, 0.0)
        agent.max_nb_steps_per_episode_scale = \
            agent.max_nb_steps_scheduler.value(episode_no - agent.learn_start_from_this_episode)
        best_eval_performance_so_far = acc
        print('Epsilon restart from epsilon {0} episode {1} with acc {2}'.format(agent.epsilon, episode_no,
                                                                                 best_eval_performance_so_far),
              file=log_file, flush=True)
        if len(running_game_points.keys()) != 0:
            running_avg_game_points = running_game_points['running_avg_game_points']
            running_avg_game_points_normalized = running_game_points['running_avg_game_points_normalized']
            running_avg_graph_rewards = running_game_points['running_avg_graph_rewards']
            running_avg_count_rewards = running_game_points['running_avg_count_rewards']
            running_avg_game_steps = running_game_points['running_avg_game_steps']
            running_avg_dqn_loss = running_game_points['running_avg_dqn_loss']
            running_avg_game_rewards = running_game_points['running_avg_game_rewards']
    print("*" * 30 + "Finish Loading Pretained Model" + "*" * 30, file=log_file, flush=True)

    # make game environments
    requested_infos = agent.select_additional_infos_lite()
    requested_infos_eval = agent.select_additional_infos()
    games_dir = "../source/dataset/"
    train_log_output_dir = './training_logs/difficulty_level_{0}'.format(agent.difficulty_level)
    json_file_name = save_to_path.split('/')[-1].replace("saved_model_dqn", "log_oorl_dqn_seed-{0}".format(args.SEED))
    if args.FIX_POINT is not None:  # load from previous training log
        save_lines = []
        if os.path.isfile(train_log_output_dir + "/" + json_file_name + '.json'):
            with open(train_log_output_dir + "/" + json_file_name + '.json', 'r') as outfile:
                training_log_lines = outfile.readlines()
                for training_log_line in training_log_lines:
                    training_log_line_dict = ast.literal_eval(training_log_line)
                    if int(training_log_line_dict['episode']) <= episode_no:
                        save_lines.append(training_log_line)
        # json_file_name = json_file_name + '_restart'
        with open(train_log_output_dir + "/" + json_file_name + '.json', 'w') as outfile:
            for save_line in save_lines:
                outfile.write(save_line)
    else:
        with open(train_log_output_dir + "/" + json_file_name + '.json', 'w') as outfile:
            outfile.close()

    # training game env
    env, _ = reinforcement_learning_dataset.get_training_game_env(games_dir + config['rl']['data_path'],
                                                                  config['rl']['difficulty_level'],
                                                                  config['rl']['training_size'],
                                                                  requested_infos,
                                                                  agent.max_nb_steps_per_episode,
                                                                  agent.batch_size,
                                                                  if_partial_game)

    if agent.run_eval:
        # training game env
        eval_env, num_eval_game = reinforcement_learning_dataset.get_evaluation_game_env(
            games_dir + config['rl']['data_path'],
            config['rl']['difficulty_level'],
            "test", # 'valid'
            requested_infos_eval,
            agent.eval_max_nb_steps_per_episode,
            agent.eval_batch_size)
    else:
        eval_env, num_eval_game = None, None

    # if os.path.exists(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt"):
    #     agent.load_pretrained_graph_generation_model(
    #         data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt")
    # else:
    #     print("No graph updater module detected... Please check ",
    #           data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt")

    while (True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(args.SEED)
        env.seed(args.SEED)
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if
                         cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        batch_size = len(obs)
        agent.train()

        game_name_list = [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list = [game.max_score for game in infos["game"]]
        chosen_prev_actions_across_goal = []
        prev_step_dones, prev_rewards = [], []
        goal_sentences_step = []
        ingredients_step = []
        goal_sentence_store_step = []

        for _ in range(batch_size):
            chosen_prev_actions_across_goal.append("restart")
            prev_step_dones.append(False)
            prev_rewards.append(0.0)
            goal_sentences, ingredients, goal_sentence_store = \
                extract_goal_sentence_from_obs(agent)

            goal_sentences_step.append(goal_sentences)
            ingredients_step.append(ingredients)
            goal_sentence_store_step.append(goal_sentence_store)
        observation_original_strings = obs
        observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite_batch(obs, infos)
        input_adjacency_matrix = agent.get_graph_adjacency_matrix(batch_size * [[]])

        # it requires to store sequences of transitions into memory with order,
        # so we use a cache to keep what agents returns, and push them into memory
        # altogether in the end of game.
        transition_cache_all = []
        still_running_mask = []
        game_rewards, game_points, graph_rewards, count_rewards = [], [], [], []
        game_rewards_by_goal = []
        print_actions = []
        selected_actions = [[] for bid in range(batch_size)]
        still_running = [1.0 - float(item) for item in prev_step_dones]
        rewards = to_pt(np.asarray([0] * agent.batch_size), enable_cuda=agent.use_cuda, type='float')

        act_randomly = episode_no < agent.learn_start_from_this_episode
        step_no = 0
        while True:
            # for step_no in range(agent.max_nb_steps_per_episode):
            observation_strings_copy = copy.copy(observation_strings)
            for bid in range(len(observation_strings_copy)):
                if " your score has just gone up by one point ." in observation_strings_copy[bid]:
                    observation_strings_copy[bid] = observation_strings_copy[bid]. \
                        replace(" your score has just gone up by one point .", "")
            with torch.no_grad():
                predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
                predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, predicted_node_mask, \
                input_node_name, input_relation_name, node_encodings, relation_encodings, \
                action_encodings_sequences, action_mask = \
                    agent.compute_updated_dynamics(input_adj_m=input_adjacency_matrix,
                                                   actions=chosen_prev_actions_across_goal,
                                                   observations=observation_strings_copy,
                                                   rewards=rewards,
                                                   hx=None,
                                                   cx=None)
            predicted_encodings = predicted_encodings_post
            new_adjacency_matrix = agent.model.decode_graph(predicted_encodings, relation_encodings)
            input_adjacency_matrix = new_adjacency_matrix

            goal_sentences_step_prev = goal_sentences_step
            ingredients_step_prev = ingredients_step
            goal_sentence_store_step_prev = goal_sentence_store_step
            goal_sentences_step = []
            ingredients_step = []
            goal_sentence_store_step = []
            for bid in range(batch_size):
                if agent.difficulty_level == 'mixed':
                    game_difficulty_level = get_game_difficulty_level(infos['game'][bid])
                else:
                    game_difficulty_level = agent.difficulty_level
                goal_sentences, ingredients, goal_sentence_store = \
                    extract_goal_sentence_from_obs(agent,
                                                   object_encodings=predicted_encodings[bid].unsqueeze(0),
                                                   object_mask=predicted_node_mask[bid].unsqueeze(0),
                                                   pre_goal_sentence=goal_sentences_step_prev[bid],
                                                   obs=observation_strings[bid],
                                                   obs_origin=observation_original_strings[bid],
                                                   ingredients=copy.copy(ingredients_step_prev[bid]),
                                                   goal_sentence_store=goal_sentence_store_step_prev[bid],
                                                   difficulty_level=game_difficulty_level,
                                                   rule_based_extraction=True)
                goal_sentences_step.append(goal_sentences)
                ingredients_step.append(ingredients)
                goal_sentence_store_step.append(goal_sentence_store)
                if len(goal_sentences) == 0:
                    still_running[bid] = 0.0

            if debug_mode:
                goal_print = []
                for bid in range(batch_size):
                    if still_running[bid]:
                        goal_print.append(goal_sentences_step[bid])
                    else:
                        goal_print.append(['END'])
                print(goal_print, file=log_file, flush=True)

            if step_no > 0:
                step_rewards_by_goal = np.zeros([num_goals, batch_size])  # compute rewards for each goal
                for bid in range(batch_size):
                    for gid in range(len(goal_sentences_step_prev[bid])):
                        if goal_sentences_step_prev[bid][gid] not in goal_sentences_step[bid]:
                            # detected goal changed, this goals have been finished, add game rewards
                            step_rewards_by_goal[gid][bid] = step_rewards[bid]
                    # if step_rewards[bid] == -1:
                    #     step_rewards_by_goal[gid][bid] = -1
                try:
                    # print_actions.append(
                    #     str({"{0} (Goal:{1}, Reward: {2})".format(chosen_actions_across_goal[0],
                    #                                               goal_sentences_pad_step[0][
                    #                                                   chosen_goal_index_across_all_goals[0]],
                    #                                               step_rewards_by_goal[
                    #                                                   chosen_goal_index_across_all_goals[0]][0]):
                    #              chosen_values_across_goal[0]}) if still_running[0] else "--")
                    print_actions.append(
                        "{0}:{1} (Goal:{2}, Reward:{3})".format(chosen_actions_across_goal[0],
                                                                chosen_values_across_goal[0],
                                                                goal_sentences_step_prev[0],
                                                                step_rewards_by_goal[:len(goal_sentences_step_prev[0]),
                                                                0],
                                                                )
                        if still_running[0] else "--")
                except:
                    print(chosen_goal_index_across_all_goals[0])
                    print(goal_sentences_pad_step[0])
                    print(step_rewards_by_goal)
                    raise ValueError("ABC")

                step_rewards_by_goal = to_pt(step_rewards_by_goal, enable_cuda=agent.use_cuda, type='float')
                game_rewards_by_goal.append(step_rewards_by_goal)

                # if all ended, break
                if np.sum(still_running) == 0:
                    break

            goal_sentence_word_ids = agent.get_goal_list_input(goal_sentences_step)
            batch_size, num_goals, goal_len = goal_sentence_word_ids.size(0), \
                                              goal_sentence_word_ids.size(1), \
                                              goal_sentence_word_ids.size(2)

            goal_sentences_pad_step = []
            goal_mask = np.zeros([batch_size, num_goals])
            for bid in range(batch_size):
                goal_sentences_pad_step.append(
                    goal_sentences_step[bid] + ['<pad>'] * (num_goals - len(goal_sentences_step[bid])))
                for gid in range(len(goal_sentences_step[bid])):
                    goal_mask[bid][gid] = 1

            chosen_actions_all_goal = []
            chosen_indices_all_goal = []
            chosen_values_all_goal = []
            force_action_indicator = np.zeros([batch_size, num_goals])
            for goal_idx in range(num_goals):  # each step has multiple goals, thus multiple choice

                force_actions = [None for bid in range(batch_size)]
                if random.uniform(0, 1) > agent.unexplore_rate:
                    force_actions = rl_force_explore(agent, infos, force_actions,
                                                     goal_sentences_pad_step, goal_sentence_store_step,
                                                     ingredients_step,
                                                     action_candidate_list, selected_actions, goal_idx, manual_explore)

                    for bid in range(batch_size):
                        if force_actions[bid] is not None:
                            force_action_indicator[bid][goal_idx] = 1

                    if debug_mode:
                        _force_candidate_action = []
                        force_flag = False
                        for bid in range(len(goal_sentences_pad_step)):
                            if force_actions[bid] is None:
                                _force_candidate_action.append(None)
                            else:
                                force_flag = True
                                _force_candidate_action.append(action_candidate_list[bid][force_actions[bid]])
                        if force_flag:
                            print('enforcing explorations: {0}'.format(_force_candidate_action),
                                  file=log_file, flush=True)

                # input_candidate_word_ids = agent.get_action_candidate_list_input(action_candidate_list)

                # generate adj_matrices
                chosen_indices, action_values, node_encodings, node_mask = \
                    agent.act_during_rl_train(node_encodings=predicted_encodings,
                                              node_mask=predicted_node_mask,
                                              action_candidate_list=action_candidate_list,
                                              input_goals_ids=goal_sentence_word_ids[:, goal_idx, :],
                                              force_actions=force_actions,
                                              random=act_randomly)
                chosen_indices_all_goal.append(chosen_indices)
                chosen_values_all_goal.append(action_values)
                chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
                chosen_actions_all_goal.append(chosen_actions)
                # tmp = [goal_sentences_pad_step[bid][goal_idx] for bid in range(batch_size)]
                # [obs_t, cmd_t, a_idx_t, graph_t, a_t-1, goal_t-1]
            force_action_mask = np.ones([batch_size, num_goals])
            force_action_label = np.sum(force_action_indicator, axis=1)
            for bid in range(batch_size):
                if force_action_label[bid] > 0:  # force exist
                    force_action_mask[bid, :] = force_action_indicator[bid, :]

            chosen_values_all_goal = np.asarray(chosen_values_all_goal)  # [goal_nums, batch_size]
            # minus the min value, so that all values are non-negative for applying the goal mask
            min_value = np.min(chosen_values_all_goal)
            chosen_values_all_goal = chosen_values_all_goal - min_value + 1e-2
            chosen_values_all_goal = chosen_values_all_goal * np.transpose(goal_mask) * np.transpose(force_action_mask)
            # masked value (zero) will never be selected
            chosen_goal_index_across_all_goals = np.argmax(chosen_values_all_goal, axis=0)

            chosen_indices_across_goal = [chosen_indices_all_goal[chosen_goal_index_across_all_goals[bid]][bid] for
                                          bid in range(batch_size)]
            chosen_actions_across_goal = [chosen_actions_all_goal[chosen_goal_index_across_all_goals[bid]][bid] for
                                          bid in range(batch_size)]
            chosen_values_across_goal = [chosen_values_all_goal[chosen_goal_index_across_all_goals[bid]][bid] for
                                         bid in range(batch_size)]
            # chosen_actions_before_parsing = [item[idx] for item, idx in
            #                                  zip(infos["admissible_commands"], chosen_indices_across_goal)]
            for bid in range(batch_size):
                selected_actions[bid].append(chosen_actions_across_goal[bid])

            transition_cache_step = []
            for goal_idx in range(num_goals):  # each step has multiple goals, thus multiple choice
                replay_info = [observation_strings, action_candidate_list, chosen_indices_across_goal,
                               to_np(predicted_encodings), to_np(predicted_node_mask),
                               chosen_prev_actions_across_goal,
                               [goal_sentences_pad_step[bid][goal_idx] for bid in range(batch_size)]]
                # observation_strings_add = []
                # action_candidate_list_add = []
                # chosen_indices_add = []
                # new_adjacency_matrix_add = []
                # chosen_prev_actions_across_goal_add = []
                # goal_sentences_add = []
                # for bid in range(batch_size):
                #     if goal_mask[bid, goal_idx]:
                #         observation_strings_add.append(observation_strings[bid])
                #         action_candidate_list_add.append(action_candidate_list[bid])
                #         chosen_indices_add.append(chosen_indices[bid])
                #         new_adjacency_matrix_add.append(new_adjacency_matrix[bid])
                #         chosen_prev_actions_across_goal_add.append(chosen_prev_actions_across_goal[bid])
                #         goal_sentences_add.append(goal_sentences_pad_step[bid][goal_idx])
                # replay_info = [observation_strings_add, action_candidate_list_add,
                #                np.asarray(chosen_indices_add), np.asarray(new_adjacency_matrix_add),
                #                chosen_prev_actions_across_goal_add, goal_sentences_add]
                transition_cache_step.append(replay_info)

            transition_cache_all.append(transition_cache_step)
            obs, scores, dones, infos = env.step(chosen_actions_across_goal)

            if debug_mode:
                print(chosen_actions_across_goal)

            chosen_prev_actions_across_goal = chosen_actions_across_goal
            # filter look and examine actions
            for bid in range(len(infos["admissible_commands"])):
                commands_ = infos["admissible_commands"][bid]
                if "examine cookbook" in selected_actions[bid]:  # prevent repeatedly exam cookbook
                    for cmd_ in [cmd for cmd in commands_ if cmd.split()[0] in ["examine", "look"]]:
                        commands_.remove(cmd_)
                else:
                    for cmd_ in [cmd for cmd in commands_ if
                                 cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                        commands_.remove(cmd_)
            observation_original_strings = obs
            observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite_batch(obs, infos)
            observation_for_counting = copy.copy(observation_strings)

            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                # agent.update_rl_models(poss_triplets_mask, episode_no, log_file)
                dqn_loss, _ = agent.update_dqn_dyna(episode_no)
                if dqn_loss is not None:
                    running_avg_dqn_loss.push(dqn_loss)
                if debug_mode:
                    parameters_info = []
                    for k, v in agent.model.named_parameters():
                        if v.grad is not None:
                            parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                        # else:
                        #     parameters_info.append("{0}:{1}".format(k, v.grad))
                    print(parameters_info, file=log_file, flush=True)

            if step_no == agent.max_nb_steps_per_episode_scale - 1:
                # terminate the game because DQN requires one extra step
                dones = [True for _ in dones]
            # if prev_step_dones != dones:
            #     print('debugging')
            step_in_total += 1
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            # step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
            step_rewards = handle_ingame_rewards(next_observations=observation_strings,
                                                 goal_sentences=goal_sentences_step,
                                                 ingredients=ingredients_step,
                                                 actions=chosen_actions_across_goal,
                                                 log_file=log_file,
                                                 apply_goal_constraint=agent.apply_goal_constraint)
            game_points.append(copy.copy([float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]))
            rewards = to_pt(np.asarray(list(scores)) - np.asarray(prev_rewards), enable_cuda=agent.use_cuda,
                            type='float')
            prev_rewards = scores

            # if agent.use_negative_reward:
            #     step_rewards = [-1.0 if _lost else r for r, _lost in
            #                     zip(step_rewards, infos["has_lost"])]  # list of float
            #     step_rewards = [5.0 if _won else r for r, _won in zip(step_rewards, infos["has_won"])]  # list of float
            step_graph_rewards = [0.0 for _ in range(batch_size)]  ## adding for obs_gen
            # counting bonus
            if agent.count_reward_lambda > 0:
                step_revisit_counting_rewards = agent.get_binarized_count(observation_for_counting, update=True)
                step_revisit_counting_rewards = [r * agent.count_reward_lambda for r in step_revisit_counting_rewards]
            else:
                step_revisit_counting_rewards = [0.0 for _ in range(batch_size)]
            still_running_mask.append(still_running)
            game_rewards.append(step_rewards)
            graph_rewards.append(step_graph_rewards)
            count_rewards.append(step_revisit_counting_rewards)
            step_no += 1

        still_running_mask_np = np.array(still_running_mask)
        game_rewards_np = np.array(game_rewards) * still_running_mask_np  # step x batch
        game_points_np = np.array(game_points) * still_running_mask_np  # step x batch
        graph_rewards_np = np.array(graph_rewards) * still_running_mask_np  # step x batch
        count_rewards_np = np.array(count_rewards) * still_running_mask_np  # step x batch
        if agent.graph_reward_lambda > 0.0:
            graph_rewards_pt = to_pt(graph_rewards_np, enable_cuda=agent.use_cuda, type='float')  # step x batch
        else:
            graph_rewards_pt = to_pt(np.zeros_like(graph_rewards_np), enable_cuda=agent.use_cuda, type='float')
        if agent.count_reward_lambda > 0.0:
            count_rewards_pt = to_pt(count_rewards_np, enable_cuda=agent.use_cuda, type='float')  # step x batch
        else:
            count_rewards_pt = to_pt(np.zeros_like(count_rewards_np), enable_cuda=agent.use_cuda, type='float')
        command_rewards_pt = to_pt(game_rewards_np, enable_cuda=agent.use_cuda, type='float')  # step x batch

        # push experience into replay buffer (dqn)
        avg_rewards_in_buffer = agent.dqn_memory.avg_rewards()
        for b in range(game_rewards_np.shape[1]):
            if still_running_mask_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_np[-1][b] != 0:
                # need to pad one transition
                _need_pad = True
                tmp_game_rewards = game_rewards_np[:, b].tolist() + [0.0]
            else:
                _need_pad = False
                tmp_game_rewards = game_rewards_np[:, b]
            if np.mean(tmp_game_rewards) < avg_rewards_in_buffer * agent.buffer_reward_threshold:
                continue
            for i in range(game_rewards_np.shape[0]):
                for gid in range(len(transition_cache_all[i])):
                    observation_strings, action_candidate_list, chosen_indices, \
                    node_encodings, node_mask, prev_action_strings, goal_sentences_pad_step = \
                        transition_cache_all[i][gid]
                    # print(getsizeof(observation_strings))
                    # print(getsizeof(triplets_pred))
                    # print(getsizeof(new_adjacency_matrix))
                    # print(getsizeof(action_candidate_list))
                    is_final = True
                    if still_running_mask_np[i][b] != 0:
                        is_final = False
                    if goal_sentences_pad_step[b] != '<pad>':
                        goal_reward = game_rewards_by_goal[i][gid][b]
                        agent.dqn_memory.add(agent.dqn_memory.data_store_step_label,
                                             observation_strings[b], goal_sentences_pad_step[b], prev_action_strings[b],
                                             action_candidate_list[b], chosen_indices[b],
                                             node_encodings[b], node_mask[b],
                                             goal_reward, graph_rewards_pt[i][b],
                                             count_rewards_pt[i][b], is_final)
                agent.dqn_memory.data_store_step_label += 1
                if still_running_mask_np[i][b] == 0:
                    break
            if _need_pad:
                for gid in range(len(transition_cache_all[-1])):
                    observation_strings, action_candidate_list, chosen_indices, \
                    node_encodings, node_mask, prev_action_strings, goal_sentences_pad_step = transition_cache_all[-1][
                        gid]
                    agent.dqn_memory.add(agent.dqn_memory.data_store_step_label,
                                         observation_strings[b], goal_sentences_pad_step[b], prev_action_strings[b],
                                         action_candidate_list[b], chosen_indices[b],
                                         node_encodings[b], node_mask[b],
                                         command_rewards_pt[-1][b] * 0.0, graph_rewards_pt[-1][b] * 0.0,
                                         count_rewards_pt[-1][b] * 0.0, True)
                agent.dqn_memory.data_store_step_label += 1

            for i in range(game_rewards_np.shape[0] - 1):
                if still_running_mask_np[i][b] == 0:  # end at i, no need to move forward
                    break
                for gid in range(len(transition_cache_all[i])):

                    observation_strings, action_candidate_list, chosen_indices, \
                    node_encodings, node_mask, \
                    prev_action_strings, goal_sentences_pad_step = transition_cache_all[i][gid]

                    next_observation_strings, _, _, \
                    _, _, _, _ = transition_cache_all[i + 1][0]
                    # we set gid = 0 since these information are shared between different goals as the same step

                    if goal_sentences_pad_step[b] != '<pad>':
                        next_goal_reward = game_rewards_by_goal[i][gid][b]
                        agent.model_memory.add(observation_strings[b], node_encodings[b], node_mask[b],
                                               goal_sentences_pad_step[b],
                                               action_candidate_list[b][chosen_indices[b]], next_goal_reward,
                                               next_observation_strings[b], )  # o, z, g, a, r', o'

        gc.collect()

        for b in range(batch_size):
            running_avg_game_points.push(np.sum(game_points_np, 0)[b])
            game_max_score_np = np.array(game_max_score_list, dtype="float32")
            running_avg_game_points_normalized.push((np.sum(game_points_np, 0) / game_max_score_np)[b])
            running_avg_game_steps.push(np.sum(still_running_mask_np, 0)[b])
            running_avg_game_rewards.push(np.sum(game_rewards_np, 0)[b])
            running_avg_graph_rewards.push(np.sum(graph_rewards_np, 0)[b])
            running_avg_count_rewards.push(np.sum(count_rewards_np, 0)[b])

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size

        # if episode_no < agent.learn_start_from_this_episode:
        #     continue
        if not debug_mode and (
                episode_no % agent.report_frequency > (episode_no - batch_size) % agent.report_frequency):
            # print(episode_no, file=log_file, flush=True)
            continue
        time_2 = datetime.datetime.now()
        memory_usage = memory_usage_psutil()
        print(
            "Episode: {:3d} | time spent: {:s} | Act by: {:s}| Epsilon {:2.5f} | Explore_rate {:2.3f} | "
            "Max step {:3.1f} | Replay Size DQN:{:3d} / Model:{:3d} | Memory Usage {:2.5f} | dqn loss: {:2.5f} | \n "
            "game points: {:2.3f} | normalized game points: {:2.3f} | "
            "game rewards: {:2.3f} | graph rewards: {:2.3f} | count rewards: {:2.3f} | used steps: {:2.3f}".format(
                episode_no, str(time_2 - time_1).rsplit(".")[0], 'Random' if act_randomly else 'Epsilon Greedy',
                agent.epsilon, 1 - agent.unexplore_rate, agent.max_nb_steps_per_episode_scale,
                len(agent.dqn_memory), agent.model_memory.length(), memory_usage, running_avg_dqn_loss.get_avg(),
                running_avg_game_points.get_avg(), running_avg_game_points_normalized.get_avg(),
                running_avg_game_rewards.get_avg(), running_avg_graph_rewards.get_avg(),
                running_avg_count_rewards.get_avg(), running_avg_game_steps.get_avg()), file=log_file, flush=True)
        print(game_name_list[0] + ":    " + " | ".join(print_actions), file=log_file, flush=True)

        # evaluate
        curr_train_performance = running_avg_game_points_normalized.get_avg()
        if agent.run_eval:  # and not debug_mode:
            now = datetime.datetime.now()
            time_store = str(now.year) + '-' + str(now.month) + '-' \
                         + str(now.day) + '-' + str(now.hour) + ':' + str(now.minute)
            planning_action_log_dir = './planning_logs/difficulty_level_{0}/' \
                                      'planning{4}_actions_df{0}_dqn_rand-{1}_{2}_seed-{5}_{3}{6}.txt'.format(
                agent.difficulty_level,
                float(agent.epsilon_anneal_to),
                save_to_path.split('/')[-1].split('_')[4],
                time_store,
                '_unsupervised' if 'unsupervised' in agent.task else '',
                args.SEED,
                debug_msg)
            planning_action_log = open(planning_action_log_dir, 'wt')
            eval_game_points, eval_game_points_normalized, eval_game_step, detailed_scores = \
                evaluate_rl_with_unsupervised_graphs(env=eval_env,
                                                     agent=agent,
                                                     num_games=num_eval_game,
                                                     debug_mode=debug_mode,
                                                     random_rate=float(agent.epsilon_anneal_to),
                                                     log_file=planning_action_log,
                                                     write_result=True)

            curr_eval_performance = eval_game_points_normalized
            # curr_performance = curr_eval_performance

            print("Saving the model with Eval performance {:2.3f}".
                  format(curr_eval_performance),
                  file=log_file,
                  flush=True)
            running_game_points = {
                'running_avg_dqn_loss': running_avg_dqn_loss,
                'running_avg_count_rewards': running_avg_count_rewards,
                'running_avg_game_points_normalized': running_avg_game_points_normalized,
                'running_avg_game_points': running_avg_game_points,
                'running_avg_game_rewards': running_avg_game_rewards,
                'running_avg_game_steps': running_avg_game_steps,
                'running_avg_graph_rewards': running_avg_graph_rewards
                                   }
            agent.save_model_to_path(save_to_path=save_to_path,
                                     episode_no=episode_no,
                                     eval_acc=curr_eval_performance,
                                     eval_loss=None,
                                     log_file=log_file)
            if curr_eval_performance > best_eval_performance_so_far:
                best_eval_performance_so_far = curr_eval_performance
                print("Saving best model so far! with Eval performance {:2.3f}".
                      format(curr_eval_performance),
                      file=log_file,
                      flush=True)
                agent.save_model_to_path(save_to_path=save_to_path.replace('.pt', '_best.pt'),
                                         episode_no=episode_no,
                                         eval_acc=curr_eval_performance,
                                         eval_loss=None,
                                         log_file=log_file,
                                         running_game_points=running_game_points,
                                         )
            else:
                if not debug_mode:
                    os.remove(planning_action_log_dir)
                    print('removing {0}'.format(planning_action_log_dir), file=log_file, flush=True)

        #     elif curr_eval_performance == best_eval_performance_so_far:
        #         if curr_eval_performance > 0.0:
        #             agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
        #         else:
        #             if curr_train_performance >= best_train_performance_so_far:
        #                 agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
        else:
            curr_eval_performance = 0.0
            detailed_scores = ""
            eval_game_points, eval_game_points_normalized, eval_game_step = 0, 0, 0
            # curr_performance = curr_train_performance
            # if curr_train_performance >= best_train_performance_so_far:
            #     agent.save_model_to_path(save_to_path=save_to_path + "/" + agent.experiment_tag + "_model.pt",
            #                              episode_no=episode_no,
            #                              eval_acc=eval_acc,
            #                              eval_loss=eval_loss,
            #                              log_file=log_file)
        # update best train performance
        if curr_train_performance >= best_train_performance_so_far:
            best_train_performance_so_far = curr_train_performance

        # if prev_performance <= curr_performance:
        #     i_am_patient = 0
        # else:
        #     i_am_patient += 1
        # prev_performance = curr_performance

        # if patient >= patience, resume from checkpoint
        # if agent.patience > 0 and i_am_patient >= agent.patience:
        #     if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
        #         print('reload from a good checkpoint...')
        #         agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt",
        #                                     load_partial_graph=False)
        #         agent.update_target_net()
        #         i_am_patient = 0

        if running_avg_game_points_normalized.get_avg() >= 0.95:
            perfect_training += 1
        else:
            perfect_training = 0

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "episode": str(episode_no),
                         "dqn loss": str(running_avg_dqn_loss.get_avg()),
                         "train game points": str(running_avg_game_points.get_avg()),
                         "train normalized game points": str(running_avg_game_points_normalized.get_avg()),
                         "train game rewards": str(running_avg_game_rewards.get_avg()),
                         "train graph rewards": str(running_avg_graph_rewards.get_avg()),
                         "train count rewards": str(running_avg_count_rewards.get_avg()),
                         "train steps": str(running_avg_game_steps.get_avg()),
                         "eval game points": str(eval_game_points),
                         "eval normalized game points": str(eval_game_points_normalized),
                         "eval steps": str(eval_game_step),
                         "detailed scores": detailed_scores})
        with open(train_log_output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()

        # if curr_performance == 1.0 and curr_train_performance >= 0.95:
        # if curr_train_performance >= 0.95:
        #     break
        # if perfect_training >= 3:
        #     break
        # if episode_no > agent.max_episode:
        #     break


if __name__ == '__main__':
    args = read_args()
    if int(args.TRAIN_FLAG):
        train(args)
    else:
        test(args)
