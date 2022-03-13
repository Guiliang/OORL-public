import ast
import datetime
import json
import os
import random
from sys import getsizeof
import copy
import numpy as np
import textworld
import torch
import yaml
import gc

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from evaluate.evaluate_rl import evaluate_rl_with_supervised_graphs
from agent.agent import OORLAgent
from generic import reinforcement_learning_dataset
from generic.data_utils import load_config, read_args, get_goal_sentence, generate_triplets_filter_mask, \
    adj_to_triplets, matching_object_from_obs, handle_ingame_rewards, process_facts, serialize_facts, \
    get_game_difficulty_level
from generic.model_utils import HistoryScoreCache, load_graph_extractor, to_pt, to_np, memory_usage_psutil


def test(args):
    raise EnvironmentError("Please use the 'test_rl_planning.py' for testing.")


def train(args):
    random.seed(args.SEED)
    time_1 = datetime.datetime.now()
    today = datetime.date.today()
    # min_unexplore_rate = 0.5
    load_extractor = True
    manual_explore = False
    neg_rewards = False
    config, debug_mode, log_file_path = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    print("Apply extractor.", file=log_file, flush=True)
    print("Apply{0} manual explore.".format('' if manual_explore else ' no'), file=log_file, flush=True)
    print("Start Training.", file=log_file, flush=True)
    debug_msg = ''  # for local machine debugging
    if debug_mode:
        # debug_msg = '_debug'
        config['rl']['epsilon_greedy']['epsilon_anneal_from'] = 0  # 0
        config['general']['training']['batch_size'] = 10
        # config['rl']['training']['max_nb_steps_per_episode'] = 50
        config['rl']['evaluate']['max_nb_steps_per_episode'] = 20
        config['rl']['training']['target_net_update_frequency'] = 50
        config['rl']['training']['learn_start_from_this_episode'] = 1
        config['rl']['replay']['update_per_k_game_steps'] = 10
        config['rl']['replay']['replay_batch_size'] = 16
        # config['rl']['replay']['replay_memory_capacity'] = 100
        # config['rl']['epsilon_greedy']['epsilon_anneal_episodes'] = 80000  # 200
        if_partial_game = False
    else:
        if_partial_game = False

    print("Max Step is {0}".format(config['rl']['evaluate']['max_nb_steps_per_episode']), flush=True, file=log_file)
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
    save_to_path = agent.output_dir + \
                   agent.experiment_tag + \
                   "/difficulty_level_{2}/saved_model_dqn_df-" \
                   "{2}-mem-{3}-epi-{4}-maxstep-{5}-anneal-{6}-{7}{8}{9}{10}{11}_{0}{1}.pt".format(
        save_date_str,
        debug_msg,
        agent.difficulty_level,
        agent.dqn_memory.capacity,
        agent.epsilon_anneal_episodes,
        agent.max_nb_steps_per_episode,
        agent.epsilon_anneal_to,
        'cstr' if agent.apply_goal_constraint else '',
        '_scratch' if not manual_explore else '',
        '_neg_reward' if neg_rewards else '',
        '-me-{0}'.format(agent.min_unexplore_rate) if agent.min_unexplore_rate is not None else '',
        '-seed-{0}'.format(args.SEED),
    )
    if args.FIX_POINT is not None:
        load_keys, episode_no, loss, acc, running_game_points = \
            agent.load_pretrained_model(load_from=save_to_path,
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

    if config['rl']['difficulty_level'] == 'mixed':
        candidate_triplets = []
        for difficulty_level in [3, 5, 7, 9]:
            set_dir = config["rl"]["triplet_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
                difficulty_level)
            with open(set_dir, 'r') as f:
                candidate_triplets_sub = f.readlines()
            candidate_triplets += candidate_triplets_sub
    else:
        set_dir = config["rl"]["triplet_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
            config['rl']['difficulty_level'])
        with open(set_dir, 'r') as f:
            candidate_triplets = f.readlines()

    filter_mask = generate_triplets_filter_mask(triplet_set=candidate_triplets,
                                                node2id=agent.node2id,
                                                relation2id=agent.relation2id)
    candidate_triplets = [triplet.replace('\n', '').split('$') for triplet in list(set(candidate_triplets))]
    poss_triplets_mask = agent.get_graph_adjacency_matrix([candidate_triplets])

    # make game environments
    requested_infos = agent.select_additional_infos_lite()
    requested_infos_eval = agent.select_additional_infos()
    games_dir = "../source/dataset/"
    train_log_output_dir = './training_logs/difficulty_level_{0}/'.format(agent.difficulty_level)
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
    if debug_mode:
        agent.max_nb_steps_per_episode = 20
        agent.max_nb_steps_per_episode_scale = 20
        agent.epsilon = 0
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
            "test",  # 'valid'
            requested_infos_eval,
            agent.eval_max_nb_steps_per_episode,
            agent.eval_batch_size)
    else:
        eval_env, num_eval_game = None, None

    if load_extractor:
        print("\n\n" + "*" * 30 + "Start Loading extractor" + "*" * 30, file=log_file, flush=True)

        if agent.difficulty_level == 'mixed':
            extractor_config_dir = '../configs/predict_graphs_dynamics_linear_seen_fineTune_df-mixed.yaml'
        else:
            extractor_config_dir = '../configs/predict_graphs_dynamics_linear_seen_fineTune_df{0}.yaml'.format(
                agent.difficulty_level)

        with open(extractor_config_dir) as reader:
            extract_config = yaml.safe_load(reader)
        extractor = OORLAgent(extract_config, log_file, '', seed=args.SEED, skip_load=True)
        load_graph_extractor(extractor, log_file)
        print("*" * 30 + "Finish Loading extractor" + "*" * 30, file=log_file, flush=True)
    else:
        extractor = None
    while (True):
        if episode_no > agent.max_episode:
            break
        env.seed(episode_no)
        np.random.seed(episode_no)
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
        print(game_name_list, file=log_file, flush=True)
        chosen_prev_actions_across_goal = []
        prev_step_dones, prev_rewards = [], []
        goal_sentences_step = []
        ingredients_step = []
        goal_sentence_store_step = []
        facts_seen_step = [set() for bid in range(batch_size)]
        for bid in range(batch_size):
            chosen_prev_actions_across_goal.append("restart")
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)
            if agent.difficulty_level == 'mixed':
                game_difficulty_level = get_game_difficulty_level(infos['game'][bid])
            else:
                game_difficulty_level = agent.difficulty_level
            goal_sentences, ingredients, goal_sentence_store = \
                get_goal_sentence(pre_goal_sentence=None,
                                  ingredients=set(),
                                  difficulty_level=game_difficulty_level)
            goal_sentences_step.append(goal_sentences)
            ingredients_step.append(ingredients)
            goal_sentence_store_step.append(goal_sentence_store)
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

        act_randomly = episode_no < agent.learn_start_from_this_episode
        step_no = 0
        while True:
            # for step_no in range(agent.max_nb_steps_per_episode):
            triplets_gtruth = []
            for bid in range(batch_size):
                facts_seen_step[bid] = process_facts(facts_seen_step[bid],
                                                     infos["game"][bid],
                                                     infos["facts"][bid],
                                                     infos["last_action"][bid],
                                                     chosen_prev_actions_across_goal[bid])
                triplets_gtruth.append(sorted(serialize_facts(facts_seen_step[bid])))
            gtruth_adjacency_matrix = extractor.get_graph_adjacency_matrix(triplets_gtruth)

            if load_extractor:
                observation_strings_copy = copy.copy(observation_strings)
                for bid in range(len(observation_strings_copy)):
                    if " your score has just gone up by one point ." in observation_strings_copy[bid]:
                        observation_strings_copy[bid] = observation_strings_copy[bid]. \
                            replace(" your score has just gone up by one point .", "")

                predicted_encodings, _, _, _, _, _, node_encodings, relation_encodings, _, _ = \
                    extractor.compute_updated_dynamics(input_adj_m=input_adjacency_matrix,  # h_t-1
                                                       actions=chosen_prev_actions_across_goal,  # a_t-1
                                                       observations=observation_strings_copy,  # o_t
                                                       hx=None,
                                                       cx=None)
                new_adjacency_matrix = extractor.model.decode_graph(predicted_encodings, relation_encodings)  # z_t, h_t
            else:
                # use posterior, it does not perform well in practice, an deterministic extractor will help
                raise ValueError('pls use an extractor')

            new_adjacency_matrix = (to_np(new_adjacency_matrix) > 0.5).astype(int)
            new_adjacency_matrix = filter_mask * new_adjacency_matrix
            triplets_pred = adj_to_triplets(adj_matrix=new_adjacency_matrix,
                                            node_vocab=extractor.node_vocab,
                                            relation_vocab=extractor.relation_vocab)
            triplets_input = adj_to_triplets(adj_matrix=input_adjacency_matrix,
                                             node_vocab=extractor.node_vocab,
                                             relation_vocab=extractor.relation_vocab)
            triplets_pred = matching_object_from_obs(observations=observation_strings,
                                                     actions=chosen_prev_actions_across_goal,
                                                     node_vocab=extractor.node_vocab,
                                                     pred_triplets=triplets_pred,
                                                     input_triplets=triplets_input)
            new_adjacency_matrix = extractor.get_graph_adjacency_matrix(triplets_pred)
            _predict_adj_matrix = to_pt(new_adjacency_matrix, extractor.use_cuda, type='float')
            input_node_name = extractor.get_graph_node_name_input()
            input_relation_name = extractor.get_graph_relation_name_input()
            predicted_encodings, _, _, predicted_node_mask = extractor.model.encode_graph(input_node_name,
                                                                                          input_relation_name,
                                                                                          _predict_adj_matrix)
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
                    get_goal_sentence(pre_goal_sentence=goal_sentences_step_prev[bid],
                                      obs=observation_strings[bid],
                                      state=np.expand_dims(gtruth_adjacency_matrix[bid], axis=0),
                                      ingredients=copy.copy(ingredients_step_prev[bid]),
                                      node2id=agent.node2id,
                                      relation2id=agent.relation2id,
                                      node_vocab=agent.node_vocab,
                                      relation_vocab=agent.relation_vocab,
                                      goal_sentence_store=goal_sentence_store_step_prev[bid],
                                      difficulty_level=game_difficulty_level,
                                      use_obs_flag=True,
                                      recheck_ingredients=False, )
                # recheck_ingredients=False if game_difficulty_level == 5 or game_difficulty_level == 1 else True)
                goal_sentences_step.append(goal_sentences)
                ingredients_step.append(ingredients)
                goal_sentence_store_step.append(goal_sentence_store)
                if len(goal_sentences) == 0:
                    still_running[bid] = 0.0

            if debug_mode:
                goal_print = []
                for bid in range(batch_size):
                    goal_print.append(goal_sentences_step[bid])
                    # if still_running[bid]:
                    #     goal_print.append(goal_sentences_step[bid])
                    # else:
                    #     goal_print.append(['END'])
                print(goal_print, file=log_file, flush=True)

            if step_no > 0:
                step_rewards_by_goal = np.zeros([num_goals, batch_size])  # compute rewards for each goal
                for bid in range(batch_size):
                    for gid in range(len(goal_sentences_step_prev[bid])):
                        if manual_explore:
                            if goal_sentences_step_prev[bid][gid] not in goal_sentences_step[bid]:
                                # detected goal changed, this goals have been finished, add game rewards
                                step_rewards_by_goal[gid][bid] = step_rewards[bid]
                        else:
                            step_rewards_by_goal[gid][bid] = step_rewards[bid]
                try:
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
                chosen_indices, action_values, node_encodings, node_mask = \
                    agent.act_during_rl_train(node_encodings=predicted_encodings,  # z_t
                                              node_mask=predicted_node_mask,  # m_t
                                              action_candidate_list=action_candidate_list,  # A_t
                                              input_goals_ids=goal_sentence_word_ids[:, goal_idx, :],  # g_t
                                              force_actions=force_actions,  # a_t
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
                                          bid in range(batch_size)]  # index of a_t
            chosen_values_across_goal = [chosen_values_all_goal[chosen_goal_index_across_all_goals[bid]][bid] for
                                         bid in range(batch_size)]
            # chosen_actions_before_parsing = [item[idx] for item, idx in
            #                                  zip(infos["admissible_commands"], chosen_indices_across_goal)]
            for bid in range(batch_size):
                selected_actions[bid].append(chosen_actions_across_goal[bid])  # a_t

            transition_cache_step = []
            for goal_idx in range(num_goals):  # each step has multiple goals, thus multiple choice
                replay_info = [observation_strings, action_candidate_list, chosen_indices_across_goal,
                               to_np(predicted_encodings), to_np(predicted_node_mask), triplets_pred,
                               chosen_prev_actions_across_goal,
                               [goal_sentences_pad_step[bid][goal_idx] for bid in range(batch_size)]]
                transition_cache_step.append(replay_info)

            transition_cache_all.append(transition_cache_step)
            obs, scores, dones, infos = env.step(chosen_actions_across_goal)

            if debug_mode:
                print(str(chosen_actions_across_goal))
                print(str(scores))
                print(str(step_no) + '\n')

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

            observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite_batch(obs, infos)
            observation_for_counting = copy.copy(observation_strings)

            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                agent.update_rl_models(poss_triplets_mask, episode_no, log_file)
                dqn_loss, _ = agent.update_dqn_dyna(episode_no, filter_mask=filter_mask)
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

            step_in_total += 1
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            # step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
            step_rewards = handle_ingame_rewards(next_observations=observation_strings,
                                                 goal_sentences=goal_sentences_step,
                                                 ingredients=ingredients_step,
                                                 goal_sentence_stores=goal_sentence_store_step,
                                                 actions=chosen_actions_across_goal,
                                                 log_file=log_file,
                                                 apply_goal_constraint=agent.apply_goal_constraint,
                                                 apply_neg_rewards=neg_rewards,
                                                 selected_actions_histories=selected_actions)
            game_points.append(copy.copy([float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]))
            prev_rewards = scores

            step_graph_rewards = [0.0 for _ in range(batch_size)]  ## adding for obs_gen
            # counting bonus
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
            # print(b)
            if still_running_mask_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_np[-1][b] != 0:
                # need to pad one transition
                _need_pad = True
                tmp_game_rewards = game_rewards_np[:, b].tolist() + [0.0]
            else:
                _need_pad = False
                tmp_game_rewards = game_rewards_np[:, b]
            # if np.mean(tmp_game_rewards) < avg_rewards_in_buffer * agent.buffer_reward_threshold:
            #     continue
            for i in range(game_rewards_np.shape[0]):
                for gid in range(len(transition_cache_all[i])):
                    observation_strings, action_candidate_list, chosen_indices, \
                    predicted_node_encodings, predicted_node_mask, triplets_pred, prev_action_strings, \
                    goal_sentences_pad_step = \
                        transition_cache_all[i][gid]
                    is_final = True
                    if still_running_mask_np[i][b] != 0:
                        is_final = False
                    if goal_sentences_pad_step[b] != '<pad>':
                        goal_reward = game_rewards_by_goal[i][gid][b]
                        # if goal_reward < 0:
                        #     print('debug')
                        agent.dqn_memory.add(agent.dqn_memory.data_store_step_label,
                                             observation_strings[b], goal_sentences_pad_step[b], prev_action_strings[b],
                                             action_candidate_list[b], chosen_indices[b],
                                             predicted_node_encodings[b], predicted_node_mask[b],
                                             goal_reward, graph_rewards_pt[i][b],
                                             count_rewards_pt[i][b], is_final)
                agent.dqn_memory.data_store_step_label += 1
                # print(still_running_mask_np[i][b])
                if still_running_mask_np[i][b] == 0:
                    # print(i)
                    break
            if _need_pad:
                for gid in range(len(transition_cache_all[-1])):
                    observation_strings, action_candidate_list, chosen_indices, \
                    predicted_node_encodings, predicted_node_mask, triplets_pred, \
                    prev_action_strings, goal_sentences_pad_step = transition_cache_all[-1][gid]
                    agent.dqn_memory.add(agent.dqn_memory.data_store_step_label,
                                         observation_strings[b], goal_sentences_pad_step[b], prev_action_strings[b],
                                         action_candidate_list[b], chosen_indices[b],
                                         predicted_node_encodings[b], predicted_node_mask[b],
                                         command_rewards_pt[-1][b] * 0.0, graph_rewards_pt[-1][b] * 0.0,
                                         count_rewards_pt[-1][b] * 0.0, True)
                agent.dqn_memory.data_store_step_label += 1

            for i in range(game_rewards_np.shape[0] - 1):
                if still_running_mask_np[i][b] == 0:  # end at i, no need to move forward
                    break
                for gid in range(len(transition_cache_all[i])):

                    observation_strings, action_candidate_list, chosen_indices, \
                    predicted_node_encodings, predicted_node_mask, triplets_pred, prev_action_strings, \
                    goal_sentences_pad_step = transition_cache_all[i][gid]

                    next_observation_strings, _, _, \
                    _, _, next_triplets_pred, _, _ = transition_cache_all[i + 1][0]
                    # we set gid = 0 since these information are shared between different goals as the same step

                    if goal_sentences_pad_step[b] != '<pad>':
                        next_goal_reward = game_rewards_by_goal[i][gid][b]
                        if next_goal_reward < 0:
                            next_goal_reward = torch.tensor(0)
                        agent.model_memory.add(observation_strings[b], triplets_pred[b], goal_sentences_pad_step[b],
                                               action_candidate_list[b][chosen_indices[b]], next_goal_reward,
                                               next_observation_strings[b], next_triplets_pred[b],
                                               )

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
        if agent.run_eval and not debug_mode:
            now = datetime.datetime.now()
            time_store = str(now.year) + '-' + str(now.month) + '-' \
                         + str(now.day) + '-' + str(now.hour) + ':' + str(now.minute)
            planning_action_log_dir = './planning_logs/difficulty_level_{0}/' \
                                      'planning{4}_actions_df{0}_dqn_rand-{1}_{2}_{3}{6}.txt'.format(
                agent.difficulty_level,
                float(agent.epsilon_anneal_to),
                save_to_path.split('/')[-1].split('_')[3],
                time_store,
                '_unsupervised' if 'unsupervised' in agent.task else '',
                args.SEED,
                debug_msg)
            planning_action_log = open(planning_action_log_dir, 'wt')
            eval_game_points, eval_game_points_normalized, eval_game_step, detailed_scores = \
                evaluate_rl_with_supervised_graphs(eval_env,
                                                   agent,
                                                   num_eval_game,
                                                   extractor,
                                                   filter_mask,
                                                   debug_mode,
                                                   # random_rate=float(agent.epsilon_anneal_to),
                                                   log_file=planning_action_log,
                                                   load_extractor=True,
                                                   write_result=True,
                                                   random_rate=0)
            curr_eval_performance = eval_game_points_normalized
            # curr_performance = curr_eval_performance
            running_game_points = {
                'running_avg_dqn_loss': running_avg_dqn_loss,
                'running_avg_count_rewards': running_avg_count_rewards,
                'running_avg_game_points_normalized': running_avg_game_points_normalized,
                'running_avg_game_points': running_avg_game_points,
                'running_avg_game_rewards': running_avg_game_rewards,
                'running_avg_game_steps': running_avg_game_steps,
                'running_avg_graph_rewards': running_avg_graph_rewards
            }
            if not debug_mode:
                agent.save_model_to_path(save_to_path=save_to_path,
                                         episode_no=episode_no,
                                         eval_acc=curr_eval_performance,
                                         eval_loss=None,
                                         log_file=log_file,
                                         running_game_points=running_game_points,
                                         )
                print("Saving the model with Eval performance {:2.3f}".
                      format(curr_eval_performance),
                      file=log_file,
                      flush=True)
            if curr_eval_performance > best_eval_performance_so_far:
                if not debug_mode:
                    print("Saving best model so far! with Eval performance {:2.3f}".
                          format(curr_eval_performance),
                          file=log_file,
                          flush=True)
                    best_eval_performance_so_far = curr_eval_performance
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
        else:
            curr_eval_performance = 0.0
            detailed_scores = ""
            eval_game_points, eval_game_points_normalized, eval_game_step = 0, 0, 0

        if running_avg_game_points_normalized.get_avg() >= 0.95:
            perfect_training += 1
        else:
            perfect_training = 0

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "episode": str(episode_no),
                         "epsilon": str(agent.epsilon),
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


if __name__ == '__main__':
    args = read_args()
    if int(args.TRAIN_FLAG):
        train(args)
    else:
        test(args)
