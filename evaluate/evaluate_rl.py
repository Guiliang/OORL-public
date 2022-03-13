import copy
import numpy as np
import torch

from generic.data_utils import get_goal_sentence, get_game_difficulty_level, process_facts, serialize_facts, \
    adj_to_triplets, matching_object_from_obs, extract_goal_sentence_from_obs
from generic.model_utils import to_np, to_pt


def evaluate_rl_with_supervised_graphs(env, agent, num_games, extractor,
                                       filter_mask, debug_mode, random_rate=0.3,
                                       log_file=None, load_extractor=True, write_result=False):
    achieved_game_points = []
    total_game_steps = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    while (True):
        if game_id >= num_games:
            break
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if
                         cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_names = [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_name_list += game_names
        game_max_score_list += [game.max_score for game in infos["game"]]
        batch_size = len(obs)
        agent.eval()

        chosen_prev_actions_across_goal, prev_game_facts = [], []
        prev_step_dones = []
        goal_sentences_step = []
        ingredients_step = []
        goal_sentence_store_step = []
        facts_seen_step = [set() for bid in range(batch_size)]
        for bid in range(batch_size):
            chosen_prev_actions_across_goal.append("restart")
            prev_game_facts.append(set())
            prev_step_dones.append(0.0)
            game_difficulty_level = get_game_difficulty_level(infos['game'][bid])
            goal_sentences, ingredients, goal_sentence_store = \
                get_goal_sentence(pre_goal_sentence=None,
                                  ingredients=set(),
                                  difficulty_level=game_difficulty_level)
            goal_sentences_step.append(goal_sentences)
            ingredients_step.append(ingredients)
            goal_sentence_store_step.append(goal_sentence_store)
        input_adjacency_matrix = agent.get_graph_adjacency_matrix(batch_size * [[]])

        observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite_batch(obs, infos)
        still_running_mask = []
        final_scores = []
        print_actions = []
        selected_actions_until_now = [[] for bid in range(batch_size)]
        selected_actions = [[] for bid in range(batch_size)]
        selected_goals = [[] for bid in range(batch_size)]
        scores_by_step = [[] for bid in range(batch_size)]
        for step_no in range(agent.eval_max_nb_steps_per_episode):

            triplets_gtruth = []
            for bid in range(batch_size):
                facts_seen_step[bid] = process_facts(facts_seen_step[bid],
                                                     infos["game"][bid],
                                                     infos["facts"][bid],
                                                     infos["last_action"][bid],
                                                     chosen_prev_actions_across_goal[bid])
                triplets_gtruth.append(sorted(serialize_facts(facts_seen_step[bid])))
            gtruth_adjacency_matrix = extractor.get_graph_adjacency_matrix(triplets_gtruth)

            # for step_no in range(5):
            if load_extractor:
                observation_strings_copy = copy.copy(observation_strings)
                for bid in range(len(observation_strings_copy)):
                    if " your score has just gone up by one point ." in observation_strings_copy[bid]:
                        observation_strings_copy[bid] = observation_strings_copy[bid]. \
                            replace(" your score has just gone up by one point .", "")

                predicted_encodings, _, _,predicted_node_mask, _, _, node_encodings, relation_encodings, _, _ = \
                    extractor.compute_updated_dynamics(input_adj_m=input_adjacency_matrix,
                                                       actions=chosen_prev_actions_across_goal,
                                                       observations=observation_strings_copy,
                                                       hx=None,
                                                       cx=None)
                new_adjacency_matrix = extractor.model.decode_graph(predicted_encodings, relation_encodings)
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
                game_difficulty_level = get_game_difficulty_level(infos['game'][bid])
                goal_sentences, ingredients, goal_sentence_store = \
                    get_goal_sentence(pre_goal_sentence=goal_sentences_step_prev[bid],
                                      obs=observation_strings[bid],
                                      state=np.expand_dims(gtruth_adjacency_matrix[bid], axis=0), #np.expand_dims(input_adjacency_matrix[bid], axis=0),
                                      ingredients=copy.copy(ingredients_step_prev[bid]),
                                      node2id=agent.node2id,
                                      relation2id=agent.relation2id,
                                      node_vocab=agent.node_vocab,
                                      relation_vocab=agent.relation_vocab,
                                      goal_sentence_store=goal_sentence_store_step_prev[bid],
                                      difficulty_level=game_difficulty_level,
                                      use_obs_flag=True,
                                      recheck_ingredients=False)
                goal_sentences_step.append(goal_sentences)
                ingredients_step.append(ingredients)
                goal_sentence_store_step.append(goal_sentence_store)
                if len(goal_sentences) == 0:
                    still_running[bid] = 0.0
            if step_no > 0:
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
            for goal_idx in range(num_goals):  # each step has multiple goals, thus multiple choice

                force_actions = [None for bid in range(batch_size)]
                chosen_indices, action_values, node_encodings, node_mask = \
                    agent.act_during_rl_test(node_encodings=predicted_encodings,
                                             node_mask=predicted_node_mask,
                                             action_candidate_list=action_candidate_list,
                                             input_goals_ids=goal_sentence_word_ids[:, goal_idx, :],
                                             force_actions=force_actions,
                                             random_rate=random_rate)
                chosen_indices_all_goal.append(chosen_indices)
                chosen_values_all_goal.append(action_values)
                chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
                chosen_actions_all_goal.append(chosen_actions)

            chosen_values_all_goal = np.asarray(chosen_values_all_goal)  # [goal_nums, batch_size]
            # minus the min value, so that all values are non-negative for applying the goal mask
            min_value = np.min(chosen_values_all_goal)
            chosen_values_all_goal = chosen_values_all_goal - min_value + 1e-2
            chosen_values_all_goal = chosen_values_all_goal * np.transpose(goal_mask)

            chosen_values_across_goal_max_indices = np.argmax(chosen_values_all_goal, axis=0)
            chosen_indices_across_goal = [chosen_indices_all_goal[chosen_values_across_goal_max_indices[bid]][bid] for
                                          bid in range(batch_size)]
            chosen_actions_across_goal = [chosen_actions_all_goal[chosen_values_across_goal_max_indices[bid]][bid] for
                                          bid in range(batch_size)]
            chosen_goals_across_goal = [goal_sentences_pad_step[bid][chosen_values_across_goal_max_indices[bid]] for
                                        bid in range(batch_size)]
            chosen_values_across_goal = [chosen_values_all_goal[chosen_values_across_goal_max_indices[bid]][bid] for
                                         bid in range(batch_size)]

            for bid in range(batch_size):
                selected_actions[bid].append(chosen_actions_across_goal[bid])  # a_t
            # send chosen actions to game engine
            chosen_actions_before_parsing = [item[idx] for item, idx in
                                             zip(infos["admissible_commands"], chosen_indices_across_goal)]

            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)

            for bid in range(batch_size):  # record results
                selected_batch_actions_until_now = [chosen_actions_across_goal[bid]] \
                    if len(selected_actions_until_now[bid]) == 0 \
                    else selected_actions_until_now[bid][-1] + [chosen_actions_across_goal[bid]]
                selected_actions_until_now[bid].append(selected_batch_actions_until_now)
                selected_goals_until_now = [chosen_goals_across_goal[bid]] if len(selected_goals[bid]) == 0 \
                    else selected_goals[bid][-1] + [chosen_goals_across_goal[bid]]
                selected_goals[bid].append(selected_goals_until_now)
                scores_by_step[bid].append(scores[bid])

            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if
                             cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look", "close"]]:
                    commands_.remove(cmd_)
            if debug_mode:
                print(goal_sentences_step)
                print(chosen_actions_across_goal)
            chosen_prev_actions_across_goal = chosen_actions_across_goal

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
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            final_scores = scores
            print_actions.append(
                str({chosen_actions_before_parsing[0]: chosen_values_across_goal[0]}) if still_running[0] else "--")
            still_running_mask.append(still_running)

        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        game_id += batch_size
    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normal_game_points = achieved_game_points / game_max_score_list

    print_strings = []
    print_strings.append("======================================================")
    print_strings.append("EVAL: rewards: {:2.3f} | normalized reward: {:2.3f} | used steps: {:2.3f}".format(
        np.mean(achieved_game_points), np.mean(normal_game_points), np.mean(total_game_steps)))
    for i in range(len(game_name_list)):
        print_strings.append(
            "game name: {}, reward: {:2.3f}, normalized reward: {:2.3f}, steps: {:2.3f}".format(game_name_list[i],
                                                                                                achieved_game_points[i],
                                                                                                normal_game_points[i],
                                                                                                total_game_steps[i]))
    print_strings.append("======================================================")
    print_strings = "\n".join(print_strings)
    print(print_strings, file=log_file, flush=True)
    return np.mean(achieved_game_points), np.mean(normal_game_points), np.mean(total_game_steps), print_strings


def evaluate_rl_with_unsupervised_graphs(env, agent, num_games, debug_mode, random_rate=0.3,
                                         log_file=None, write_result=False):
    achieved_game_points = []
    total_game_steps = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    if debug_mode:
        print("\n\nBegin testing")
    while (True):
        if game_id >= num_games:
            break
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if
                         cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_names = [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_name_list += game_names
        game_max_score_list += [game.max_score for game in infos["game"]]
        batch_size = len(obs)
        agent.eval()

        chosen_actions_before_parsing = batch_size * ['restart']

        chosen_prev_actions_across_goal, prev_game_facts = [], []
        prev_step_dones = []
        prev_rewards = []
        goal_sentences_step = []
        ingredients_step = []
        goal_sentence_store_step = []
        for _ in range(batch_size):
            chosen_prev_actions_across_goal.append("restart")
            prev_game_facts.append(set())
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)
            goal_sentences, ingredients, goal_sentence_store = \
                extract_goal_sentence_from_obs(agent)
            goal_sentences_step.append(goal_sentences)
            ingredients_step.append(ingredients)
            goal_sentence_store_step.append(goal_sentence_store)
        input_adjacency_matrix = agent.get_graph_adjacency_matrix(batch_size * [[]])

        observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite_batch(obs, infos)
        still_running_mask = []
        final_scores = []
        print_actions = []
        selected_actions = [[] for bid in range(batch_size)]
        selected_goals = [[] for bid in range(batch_size)]
        scores_by_step = [[] for bid in range(batch_size)]
        rewards = to_pt(np.asarray([0] * batch_size), enable_cuda=agent.use_cuda, type='float')
        for step_no in range(agent.eval_max_nb_steps_per_episode):
            # for step_no in range(5):
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
                game_difficulty_level = get_game_difficulty_level(infos['game'][bid])
                goal_sentences, ingredients, goal_sentence_store = \
                    extract_goal_sentence_from_obs(agent,
                                                   object_encodings=predicted_encodings[bid].unsqueeze(0),
                                                   object_mask=predicted_node_mask[bid].unsqueeze(0),
                                                   pre_goal_sentence=goal_sentences_step_prev[bid],
                                                   obs=observation_strings[bid],
                                                   obs_origin=None,
                                                   ingredients=copy.copy(ingredients_step_prev[bid]),
                                                   goal_sentence_store=goal_sentence_store_step_prev[bid],
                                                   difficulty_level=game_difficulty_level,
                                                   rule_based_extraction=False)
                goal_sentences_step.append(goal_sentences)
                ingredients_step.append(ingredients)
                goal_sentence_store_step.append(goal_sentence_store)
                if len(goal_sentences) == 0:
                    still_running[bid] = 0.0

            if debug_mode:
                print(chosen_actions_before_parsing)
                print(goal_sentences_step)

            if step_no > 0:
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
            for goal_idx in range(num_goals):  # each step has multiple goals, thus multiple choice

                force_actions = [None for bid in range(batch_size)]
                # force opening fridge to reveal items in fridge
                for bid in range(len(goal_sentences_pad_step)):
                    if 'open fridge' in action_candidate_list[bid] and 'open fridge' not in selected_actions[bid]:
                        force_actions[bid] = action_candidate_list[bid].index('open fridge')

                # input_candidate_word_ids = agent.get_action_candidate_list_input(action_candidate_list)
                chosen_indices, action_values, node_encodings, node_mask = \
                    agent.act_during_rl_test(node_encodings=predicted_encodings,
                                             node_mask=predicted_node_mask,
                                             action_candidate_list=action_candidate_list,
                                             input_goals_ids=goal_sentence_word_ids[:, goal_idx, :],
                                             force_actions=force_actions,
                                             random_rate=random_rate)
                chosen_indices_all_goal.append(chosen_indices)
                chosen_values_all_goal.append(action_values)
                chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
                chosen_actions_all_goal.append(chosen_actions)

            chosen_values_all_goal = np.asarray(chosen_values_all_goal)  # [goal_nums, batch_size]
            # minus the min value, so that all values are non-negative for applying the goal mask
            min_value = np.min(chosen_values_all_goal)
            chosen_values_all_goal = chosen_values_all_goal - min_value + 1e-2
            chosen_values_all_goal = chosen_values_all_goal * np.transpose(goal_mask)

            chosen_values_across_goal_max_indices = np.argmax(chosen_values_all_goal, axis=0)
            chosen_indices_across_goal = [chosen_indices_all_goal[chosen_values_across_goal_max_indices[bid]][bid] for
                                          bid in range(batch_size)]
            chosen_actions_across_goal = [chosen_actions_all_goal[chosen_values_across_goal_max_indices[bid]][bid] for
                                          bid in range(batch_size)]
            chosen_goals_across_goal = [goal_sentences_pad_step[bid][chosen_values_across_goal_max_indices[bid]] for
                                        bid in range(batch_size)]
            chosen_values_across_goal = [chosen_values_all_goal[chosen_values_across_goal_max_indices[bid]][bid] for
                                         bid in range(batch_size)]
            # send chosen actions to game engine
            chosen_actions_before_parsing = [item[idx] for item, idx in
                                             zip(infos["admissible_commands"], chosen_indices_across_goal)]

            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)

            rewards = to_pt(np.asarray(list(scores)) - np.asarray(prev_rewards), enable_cuda=agent.use_cuda,
                            type='float')
            prev_rewards = scores

            for bid in range(batch_size):  # record results
                selected_actions_until_now = [chosen_actions_across_goal[bid]] if len(selected_actions[bid]) == 0 \
                    else selected_actions[bid][-1] + [chosen_actions_across_goal[bid]]
                selected_actions[bid].append(selected_actions_until_now)
                selected_goals_until_now = [chosen_goals_across_goal[bid]] if len(selected_goals[bid]) == 0 \
                    else selected_goals[bid][-1] + [chosen_goals_across_goal[bid]]
                selected_goals[bid].append(selected_goals_until_now)
                scores_by_step[bid].append(scores[bid])

            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if
                             cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look", "close"]]:
                    commands_.remove(cmd_)

            chosen_prev_actions_across_goal = chosen_actions_across_goal
            observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite_batch(obs, infos)
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            final_scores = scores
            print_actions.append(
                str({chosen_actions_before_parsing[0]: chosen_values_across_goal[0]}) if still_running[0] else "--")
            still_running_mask.append(still_running)

        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        game_id += batch_size
    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normal_game_points = achieved_game_points / game_max_score_list

    print_strings = []
    print_strings.append("======================================================")
    print_strings.append("EVAL: rewards: {:2.3f} | normalized reward: {:2.3f} | used steps: {:2.3f}".format(
        np.mean(achieved_game_points), np.mean(normal_game_points), np.mean(total_game_steps)))
    for i in range(len(game_name_list)):
        print_strings.append(
            "game name: {}, reward: {:2.3f}, normalized reward: {:2.3f}, steps: {:2.3f}".format(game_name_list[i],
                                                                                                achieved_game_points[i],
                                                                                                normal_game_points[i],
                                                                                                total_game_steps[i]))
    print_strings.append("======================================================")
    print_strings = "\n".join(print_strings)
    print(print_strings, file=log_file, flush=True)
    return np.mean(achieved_game_points), np.mean(normal_game_points), np.mean(total_game_steps), print_strings