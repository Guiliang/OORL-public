import copy
import datetime
import json
import os
import math
import torch
import numpy as np
import torch.nn.functional as F

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from generic.plot_utils import plot_heatmap
from evaluate.evaluate_prediction import evaluate_reward_predictor_rnn, test_on_textworld_unsupervised
from generic.reward_prediction_dataset import RewardPredictionDynamicDataRNN, RewardPredictionDataRNN
from generic.state_prediction_dataset import SPDataUnSupervised
from agent.agent import OORLAgent
from generic.data_utils import load_config, read_args, handle_rnn_max_len, merge_sample_triplet_index, \
    get_goal_sentence, extract_goal_sentence_from_obs
from generic.model_utils import HistoryScoreCache, to_np, to_pt


def test(args):
    import textworld
    from generic.data_utils import process_facts, serialize_facts
    config, debug_mode, log_file_path, model_name = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

    agent_planning = OORLAgent(config, log_file, '', skip_load=True)

    load_path = agent_planning.output_dir + "/reward_predictor/difficulty_level_{5}" \
                                            "/saved_model_{4}_dynamic_linear_{3}_{6}{0}_" \
                                            "df-{5}_weight-{7}{8}_{1}{2}.pt".format(
        agent_planning.model.graph_decoding_method,
        'obs_loss_Aug-17-2021',  # obs_loss_Aug-17-2021
        '',
        agent_planning.model.dynamic_model_mechanism,
        agent_planning.model.dynamic_loss_type,
        agent_planning.difficulty_level,
        "cond_dec-" if agent_planning.if_condition_decoder else "dec-",
        0.2,
        "",  # _multi_opt
    )
    # df-7_weight-0.2_Jul-09-2021.

    agent_planning.load_pretrained_model(load_path, log_file=log_file, load_partial_graph=False)

    if agent_planning.difficulty_level == 'mixed':
        game_difficulty_level = 7
    else:
        game_difficulty_level = agent_planning.difficulty_level
    if agent_planning.difficulty_level == 'mixed':
        test_path = '../source/dataset/rl.0.2/test/difficulty_level_{0}/'.format(game_difficulty_level)
    else:
        test_path = '../source/dataset/rl.0.2/test/difficulty_level_{0}/'.format(agent_planning.difficulty_level)
    test_files = []
    for file in os.listdir(test_path):
        if file.endswith('.z8'):
            test_files.append(file)
    test_files = [test_files[1]]
    for test_file in test_files:
        # for test_file in ['tw-cooking-recipe1+take1+cut+open-Y8oktyOLCkBLTglqh7pJN.z8']:
        if test_file.endswith('.json'):
            continue
        game_file = test_path + test_file
        print(game_file)
        agent_planning.use_negative_reward = False
        eval_requested_infos = agent_planning.select_additional_infos_lite()
        eval_requested_infos.extras = ["walkthrough"]
        env = textworld.start(game_file, infos=eval_requested_infos)
        env = textworld.envs.wrappers.Filter(env)

        # def _test_on_textworld(input_adj_m, action, obs, reward, goal_sentences, ingredients, goal_sentence_store,
        #                        hx, cx, _last_facts, infos):
        #
        #     observation_string, action_candidate_list = agent_planning.get_game_info_at_certain_step_lite(obs, infos)
        #
        #     if " your score has just gone up by one point ." in observation_string:
        #         observation_string = observation_string.replace(" your score has just gone up by one point .", "")
        #     observations = [observation_string]
        #     actions = [action]
        #
        #     _facts_seen = process_facts(_last_facts,
        #                                 infos["game"],
        #                                 infos["facts"],
        #                                 infos["last_action"],
        #                                 action)
        #
        #     predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
        #     predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
        #     input_node_name, input_relation_name, node_encodings, relation_encodings, \
        #     _, _ = \
        #         agent_planning.compute_updated_dynamics(input_adj_m=input_adj_m,
        #                                                 actions=actions,
        #                                                 observations=observations,
        #                                                 goal_sentences=None,
        #                                                 rewards=reward,
        #                                                 hx=hx,
        #                                                 cx=cx, )
        #
        #     goal_sentences_next, ingredients, goal_sentence_store = \
        #         extract_goal_sentence_from_obs(
        #             agent=agent_planning,
        #             object_encodings=predicted_encodings_post,
        #             object_mask=node_mask,
        #             pre_goal_sentence=goal_sentences,
        #             obs=observation_string,
        #             obs_origin=obs,
        #             ingredients=copy.copy(ingredients),
        #             goal_sentence_store=goal_sentence_store,
        #             difficulty_level=game_difficulty_level,
        #             rule_based_extraction=False)
        #
        #     predicted_encodings = predicted_encodings_post
        #     predict_output_adj = agent_planning.model.decode_graph(predicted_encodings, relation_encodings)
        #     # tmp = to_np(predict_output_adj[0, 2, :, :])
        #     # predict_heatmap_matrix = np.mean(to_np(predict_output_adj), axis=1)[0, :, :]
        #     # plot_heatmap(tmp)
        #     # plot_heatmap(to_np(predict_output_adj)[0, 2, :, :])
        #     dim = 1
        #     predict_heatmap_matrix = to_np(predict_output_adj)[0, dim, :, :]
        #     plot_heatmap(predict_heatmap_matrix,
        #                  plot_name="./plot_results_figures/heatmap/heatmap-unsupervised-step-{0}-idx{1}".format(i, dim),
        #                  title="$\hat{h}_{t}$ learned by SS-ELBo",
        #                  add_bar=True,
        #                  fig_size=(6, 5)
        #                  )
        #
        #     input_actions = agent_planning.get_word_input(actions, minimum_len=10)  # batch x action_len
        #     action_encodings_sequences, action_mask = \
        #         agent_planning.model.encode_text_for_reward_prediction(input_actions)
        #
        #     print("Action: '{0}', \nObservation: '{1}'".format(action, observation_string))
        #     for goal_sentence in goal_sentences:
        #         input_goals = agent_planning.get_word_input([goal_sentence], minimum_len=20)  # batch x goal_len
        #         goal_encodings_sequences, goal_mask = \
        #             agent_planning.model.encode_text_for_reward_prediction(input_goals)
        #
        #         pred_rewards = agent_planning.compute_rewards_unsupervised(
        #             predicted_encodings=predicted_encodings,
        #             node_mask=node_mask,
        #             action_encodings_sequences=action_encodings_sequences,
        #             action_mask=action_mask,
        #             goal_encodings_sequences=goal_encodings_sequences,
        #             goal_mask=goal_mask)
        #         pred_rewards = F.softmax(pred_rewards, dim=1)
        #         pred_rewards_label = 1 - torch.argmax(pred_rewards).item()
        #         print("Goal: [{0}] with reward: {1} and confidence: {2}.".format(goal_sentence,
        #                                                                          pred_rewards_label,
        #                                                                          pred_rewards[0][
        #                                                                              1 - pred_rewards_label].item()))
        #     print('')
        #
        #     return predict_output_adj, action_candidate_list, _facts_seen, \
        #            goal_sentences_next, ingredients, goal_sentence_store

        _last_facts = set()
        obs, infos = env.reset()
        walkthrough = infos["extra.walkthrough"]
        input_adj_m = np.zeros((1,
                                len(agent_planning.relation_vocab),
                                len(agent_planning.node_vocab),
                                len(agent_planning.node_vocab)),
                               dtype="float32")

        goal_sentences, ingredients, goal_sentence_store = \
            extract_goal_sentence_from_obs(
                agent=agent_planning,
                object_encodings=None,
                object_mask=None,
                pre_goal_sentence=None,
                obs=None,
                obs_origin=None,
                ingredients=None,
                goal_sentence_store=None,
                difficulty_level=agent_planning.difficulty_level)
        action = 'restart'
        hx = None
        cx = None
        done = 0
        # reward = to_pt(np.asarray([0]), enable_cuda=agent_planning.use_cuda, type='float')
        # while not done:
        for i in range(len(walkthrough) + 1):
            # for i in range(50):
            input_adj_m, action_candidate_list, _last_facts, \
            goal_sentences, ingredients, goal_sentence_store, \
            save_object_prior_embeddings, save_object_posterior_embeddings \
                = test_on_textworld_unsupervised(
                agent_planning=agent_planning,
                game_difficulty_level=game_difficulty_level,
                input_adj_m=input_adj_m,
                action=action,
                obs=obs,
                # reward=reward,
                goal_sentences=goal_sentences,
                ingredients=ingredients,
                goal_sentence_store=goal_sentence_store,
                hx=hx,
                cx=cx,
                _last_facts=_last_facts,
                infos=infos,
                action_counter=i
            )
            if i < len(walkthrough):
                action = walkthrough[i]
            else:
                action_idx = np.random.choice(len(action_candidate_list))
                action = action_candidate_list[action_idx]
            obs, scores, done, infos = env.step(action)
        break


def train(args):
    time_1 = datetime.datetime.now()
    today = datetime.date.today()
    config, debug_mode, log_file_path = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

    # print('-' * 100, file=log_file, flush=True)
    # print("*** Warning: Launching the debugging with detach on rewards prediction and constraint latent loss ***",
    #       file=log_file, flush=True)
    # # config['general']['training']['fix_parameters_keywords'] = ["decode_graph", "dynamic_model", "rgcns", "word_embedding", "node_embedding", "relation_embedding", "word_embedding_prj"]
    # print('-' * 100, file=log_file, flush=True)

    apply_history_reward_loss = True
    split_optimizer = False
    skip_obs_loss = False

    debug_msg = ''
    eval_max_counter = None
    max_len_force = None

    print('{0} obs loss'.format('Skip' if skip_obs_loss else 'Apply'), file=log_file, flush=True)
    print("Apply additional text encoder", file=log_file, flush=True)
    print("Apply{0} history rewards".format('' if apply_history_reward_loss else ' no'), file=log_file, flush=True)
    print("Apply {0} optimizer(s)".format('multiple' if split_optimizer else 'one'), file=log_file, flush=True)

    if debug_mode:
        debug_msg = '_debug'
        # config['general']['use_cuda'] = False
        config['general']['training']['batch_size'] = 4
        config['general']['checkpoint']['report_frequency'] = 1
        config['general']['training']['optimizer']['learning_rate'] = 0.0005
        config['general']['training']['optimizer']['learning_rate_warmup_until'] = 1000
        # config['general']['checkpoint']['report_frequency'] = 1
        eval_max_counter = 2
    print("Debug mode is {0}".format(debug_mode), file=log_file, flush=True)
    agent = OORLAgent(config=config, log_file=log_file, debug_msg=debug_msg, split_optimizer=split_optimizer)

    if agent.difficulty_level == 'mixed':
        goal_extract_dict = {}
        for difficulty_level in [3, 5, 7, 9]:
            goal_extraction_dataset = '../source/dataset/gt.0.1/difficulty_level_{0}/' \
                                      'goal_extraction_dict.json'.format(difficulty_level)
            print("Loading goal extraction dataset from {0}".format(goal_extraction_dataset),
                  file=log_file, flush=True)
            with open(goal_extraction_dataset) as goal_extract_file:
                goal_extract_sub_dict = json.loads(goal_extract_file.read())
            for key in goal_extract_sub_dict.keys():
                if key not in goal_extract_dict.keys():
                    goal_extract_dict.update({key: goal_extract_sub_dict[key]})
    else:
        goal_extraction_dataset = '../source/dataset/gt.0.1/difficulty_level_{0}/' \
                                  'goal_extraction_dict.json'.format(agent.difficulty_level)
        print("Loading goal extraction dataset from {0}".format(goal_extraction_dataset),
              file=log_file, flush=True)
        with open(goal_extraction_dataset) as goal_extract_file:
            goal_extract_dict = json.loads(goal_extract_file.read())

    # env = SPDataUnSupervised(config, log_file)
    env = RewardPredictionDynamicDataRNN(config, agent, log_file=log_file)
    # env = RewardPredictionDataRNN(config, log_file)
    env.split_reset("train")
    ave_train_loss = HistoryScoreCache(capacity=500)
    ave_train_reward_loss = HistoryScoreCache(capacity=500)
    ave_train_latent_loss = HistoryScoreCache(capacity=500)
    ave_train_goal_loss = HistoryScoreCache(capacity=500)
    ave_train_obs_loss = HistoryScoreCache(capacity=500)
    ave_train_reward_precision = HistoryScoreCache(capacity=500)
    ave_train_reward_recall = HistoryScoreCache(capacity=500)
    ave_train_reward_f1 = HistoryScoreCache(capacity=500)
    ave_train_reward_acc = HistoryScoreCache(capacity=500)
    ave_valid_reward_precision = HistoryScoreCache(capacity=500)
    ave_valid_reward_recall = HistoryScoreCache(capacity=500)
    ave_valid_reward_f1 = HistoryScoreCache(capacity=500)
    ave_valid_reward_acc = HistoryScoreCache(capacity=500)
    ave_valid_reward_soft_dist = HistoryScoreCache(capacity=500)
    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        loss_win = None
        eval_acc_win = None
        viz_loss, viz_eval_loss, viz_eval_acc = [], [], []
    # debug_mode = False
    episode_no = 0
    batch_no = 0
    best_f1_so_far = 0
    weight_init = 0
    if 'latent' in agent.model.dynamic_loss_type:
        weight_upper_bound = 0.2
    else:
        weight_upper_bound = weight_init
    learning_rate = config['general']['training']['optimizer']['learning_rate']

    if args.FIX_POINT is None:
        save_date_str = today.strftime("%b-%d-%Y")
    else:
        save_date_str = args.FIX_POINT

    save_to_path = agent.output_dir + agent.experiment_tag + \
                   "/difficulty_level_{5}/saved_model_{4}_dynamic_linear_{3}_{6}{0}_" \
                   "df-{5}_weight-{7}{8}{9}{10}_{1}{2}.pt".format(
                       agent.model.graph_decoding_method,
                       save_date_str,
                       debug_msg,
                       agent.model.dynamic_model_mechanism,
                       agent.model.dynamic_loss_type,
                       env.difficulty_level,
                       "cond_dec-" if agent.if_condition_decoder else "dec-",
                       weight_upper_bound,
                       "_dropout" if agent.if_reward_dropout else '',
                       "_multi_opt" if split_optimizer else '',
                       '_obs_loss' if not skip_obs_loss else ''
                   )
    # current_triplets, previous_triplets, current_observations, previous_actions, \
    # current_rewards, current_goal_sentences = env.get_batch()
    # print('-' * 100, file=log_file, flush=True)
    # print("*** Warning: Launching the overfitting experiment with few samples. ***", file=log_file, flush=True)
    # print('-' * 100, file=log_file, flush=True)

    if args.FIX_POINT is not None:
        load_keys, episode_no, loss, acc = agent.load_pretrained_model(load_from=save_to_path, log_file=log_file)
        batch_no = int(episode_no / agent.batch_size)
        best_f1_so_far = acc
        print('Epsilon restart from episode {0}/ batch {1} with acc {2}'.format(episode_no, batch_no, best_f1_so_far),
              file=log_file, flush=True)

    try:
        while (True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            # current_triplets, previous_triplets, current_observations, previous_actions, current_rewards, \
            current_triplets, previous_triplets, current_observations, previous_actions, \
            current_rewards, current_goal_sentences = env.get_batch()
            # print(current_goal_sentences, file=log_file, flush=True)
            curr_batch_size = len(current_observations)
            origin_lens = [len(elem) for elem in current_observations]
            if max_len_force is not None:
                max_len = max(origin_lens) if max(origin_lens) <= max_len_force else max_len_force
            else:
                max_len = max(origin_lens)

            previous_actions, lens = handle_rnn_max_len(max_len, previous_actions, "<pad>")
            current_observations, lens = handle_rnn_max_len(max_len, current_observations, "<pad>")
            current_rewards, lens = handle_rnn_max_len(max_len, current_rewards, 0)
            current_goal_sentences, lens = handle_rnn_max_len(max_len, current_goal_sentences, "<pad>")

            batch_masks = torch.zeros((curr_batch_size, max_len), dtype=torch.float)
            if agent.use_cuda:
                batch_masks = batch_masks.cuda()
            if apply_history_reward_loss:
                for i in range(curr_batch_size):
                    batch_masks[i, :lens[i]] = 1
            else:
                for i in range(curr_batch_size):
                    batch_masks[i, lens[i] - 1] = 1

            obs_preds_last_batch = []
            goal_preds_obs_batch = []
            goal_preds_tf_batch = []
            goal_preds_greedy_words_batch = []
            last_k_batches_loss = []
            last_k_batches_obs_loss = []
            last_k_batches_reward_loss = []
            last_k_batches_latent_loss = []
            last_k_batches_goal_gen_loss = []
            prev_h = None
            input_adj_t = None

            train_tp_num_reward, train_tn_num_reward, \
            train_fp_num_reward, train_fn_num_reward = 0, 0, 0, 0

            curr_load_data_batch_size = len(current_observations)
            episode_no += curr_load_data_batch_size
            batch_no += 1  # applying learning rate warmup

            for i in range(max_len):
                current_t_observations = [elem[i] for elem in current_observations]
                action_t = [elem[i] for elem in previous_actions]
                rewards_t = [elem[i] for elem in current_rewards]
                current_t_goal_sentences = [elem[i] for elem in current_goal_sentences]

                current_goal_extraction_targets_t = []
                goal_extraction_masks_t = torch.zeros((curr_batch_size), dtype=torch.float)
                for bid in range(curr_batch_size):
                    current_t_observation = current_t_observations[bid]
                    find_flag = False
                    for key in goal_extract_dict.keys():
                        if key in current_t_observation:
                            current_goal_extraction_targets_t.append(goal_extract_dict[key])
                            find_flag = True
                            goal_extraction_masks_t[bid] = 1
                            break
                    if not find_flag:
                        current_goal_extraction_targets_t.append('<pad>')
                if agent.use_cuda:
                    goal_extraction_masks_t = goal_extraction_masks_t.cuda()

                # target_t_graph_triplets = [elem[i] for elem in target_graph_triplets]
                # previous_t_graph_triplets = [elem[i] for elem in previous_graph_triplets]
                # input_t_adjacency_matrix = agent.get_graph_adjacency_matrix(previous_t_graph_triplets)
                # output_t_adjacency_matrix = agent.get_graph_adjacency_matrix(target_t_graph_triplets)

                if input_adj_t is None:
                    input_adj_t = np.zeros(
                        (curr_batch_size, len(agent.relation_vocab), len(agent.node_vocab), len(agent.node_vocab)),
                        dtype="float32")

                obs_gen_loss_t, predict_adj_t, obs_pred_t, \
                r_loss_t, pred_t_rewards, real_t_rewards, \
                latent_loss_t, \
                goal_gen_loss_t, goal_pred_tf_t, goal_pred_greedy_words_t = \
                    agent.get_unsupervised_dynamics_logistic(
                        input_adj_m=input_adj_t,
                        actions=action_t,
                        observations=current_t_observations,
                        real_rewards=rewards_t,
                        goal_sentences=current_t_goal_sentences,
                        batch_masks=batch_masks[:, i],
                        goal_extraction_targets=current_goal_extraction_targets_t,
                        goal_extraction_masks=goal_extraction_masks_t,
                    )
                input_adj_t = predict_adj_t
                if debug_mode:
                    weight = weight_upper_bound
                else:
                    # weight = weight_init + float(weight_upper_bound - weight_init) * batch_no / (agent.warmup_until) \
                    #     if batch_no <= agent.warmup_until else weight_upper_bound
                    weight = weight_upper_bound
                if torch.sum(batch_masks[:, i]) > 0:
                    if skip_obs_loss:
                        last_k_batches_loss.append(r_loss_t + weight * latent_loss_t)
                    else:
                        last_k_batches_loss.append(r_loss_t + weight * latent_loss_t + obs_gen_loss_t)
                    # last_k_batches_loss.append(r_loss_t)
                    # last_k_batches_loss.append(latent_loss_t)
                    last_k_batches_reward_loss.append(r_loss_t)
                    last_k_batches_obs_loss.append(obs_gen_loss_t)
                    last_k_batches_latent_loss.append(weight * latent_loss_t)
                    # obs_preds_last_batch.append(obs_pred_t[-1])
                for bid in range(curr_batch_size):
                    apply_goal_gen_loss = False
                    if goal_extraction_masks_t[bid]:
                        obs_preds_last_batch.append(obs_pred_t[bid])  # TODO: it was -1
                        goal_preds_obs_batch.append(current_t_observations[bid])
                        goal_preds_tf_batch.append(goal_pred_tf_t[bid])
                        goal_preds_greedy_words_batch.append(goal_pred_greedy_words_t[bid])
                        apply_goal_gen_loss = True
                    if apply_goal_gen_loss:
                        last_k_batches_goal_gen_loss.append(goal_gen_loss_t)

                """Reward Accuracy"""
                for idx in range(len(pred_t_rewards)):
                    if batch_masks[idx, i]:
                        pred_ = 1 - np.argmax(pred_t_rewards[idx])  # idx 0 is positive, idx 1 is negative
                        real_ = 1 - np.argmax(real_t_rewards[idx])  # idx 0 is positive, idx 1 is negative

                        if pred_ == 1 and real_ == 1:
                            train_tp_num_reward += 1
                        elif pred_ == 1 and real_ == 0:
                            train_fp_num_reward += 1
                        elif pred_ == 0 and real_ == 1:
                            train_fn_num_reward += 1
                        elif pred_ == 0 and real_ == 0:
                            train_tn_num_reward += 1
                        else:
                            print(batch_masks[:, i], file=log_file, flush=True)
                            print(pred_t_rewards)
                            print(real_t_rewards)
                            raise ("ROC Mistake")

            if split_optimizer:
                agent.model.zero_grad()

                agent.optimizers['transition_model'].zero_grad()
                ave_k_latent_loss = torch.mean(torch.stack(last_k_batches_latent_loss))
                ave_k_latent_loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                agent.optimizers['transition_model'].step()  # apply gradients

                agent.optimizers['reward_model'].zero_grad()
                ave_k_reward_loss = torch.mean(torch.stack(last_k_batches_reward_loss))
                ave_k_reward_loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                agent.optimizers['reward_model'].step()  # apply gradients

            else:
                agent.model.zero_grad()
                agent.optimizer.zero_grad()
                ave_k_loss = torch.mean(torch.stack(last_k_batches_loss))
                ave_k_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                agent.optimizer.step()

            if debug_mode:
                parameters_info = []
                for k, v in agent.model.named_parameters():
                    if v.grad is not None:
                        parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                    else:
                        parameters_info.append("{0}:{1}".format(k, v.grad))
                print(parameters_info, file=log_file, flush=True)

            agent.goal_gen_optimizer.zero_grad()
            if len(last_k_batches_goal_gen_loss) > 0:
                ave_k_goal_gen_loss = torch.mean(torch.stack(last_k_batches_goal_gen_loss))
                ave_k_goal_gen_loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
                agent.goal_gen_optimizer.step()  # apply gradients
                ave_train_goal_loss.push(to_np(ave_k_goal_gen_loss))
            if debug_mode:
                parameters_info = []
                for k, v in agent.model.named_parameters():
                    if v.grad is not None:
                        parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                    else:
                        parameters_info.append("{0}:{1}".format(k, v.grad))
                print(parameters_info, file=log_file, flush=True)

            ave_k_loss = torch.mean(torch.stack(last_k_batches_loss))
            ave_train_loss.push(to_np(ave_k_loss))
            ave_k_reward_loss = torch.mean(torch.stack(last_k_batches_reward_loss))
            ave_train_reward_loss.push(to_np(ave_k_reward_loss))
            ave_k_obs_loss = torch.mean(torch.stack(last_k_batches_obs_loss))
            ave_train_obs_loss.push(to_np(ave_k_obs_loss))
            ave_k_latent_loss = torch.mean(torch.stack(last_k_batches_latent_loss))
            ave_train_latent_loss.push(to_np(ave_k_latent_loss))

            if train_tp_num_reward + train_fp_num_reward > 0:
                train_precision_reward = float(train_tp_num_reward) / (
                        train_tp_num_reward + train_fp_num_reward)
            else:
                train_precision_reward = 0
            if train_tp_num_reward + train_fn_num_reward > 0:
                train_recall_reward = float(train_tp_num_reward) / (
                        train_tp_num_reward + train_fn_num_reward)
            else:
                train_recall_reward = 0
            if train_tp_num_reward + train_tn_num_reward + train_fp_num_reward + train_fn_num_reward == 0:
                print(batch_masks, file=log_file, flush=True)
                raise ("ROC Mistake")

            train_acc_reward = float(train_tp_num_reward + train_tn_num_reward) / (
                    train_tp_num_reward + train_tn_num_reward + train_fp_num_reward +
                    train_fn_num_reward)
            if train_recall_reward + train_precision_reward > 0:
                train_f1_reward = 2 * train_precision_reward * train_recall_reward / (
                        train_recall_reward + train_precision_reward)
            else:
                train_f1_reward = 0

            ave_train_reward_precision.push(train_precision_reward)
            ave_train_reward_recall.push(train_recall_reward)
            ave_train_reward_f1.push(train_f1_reward)
            ave_train_reward_acc.push(train_acc_reward)
            time_2 = datetime.datetime.now()

            # if True:
            if episode_no % agent.report_frequency <= (
                    episode_no - curr_batch_size) % agent.report_frequency or debug_mode and agent.run_eval:

                # k = 0
                # ep_string_obs_tf = []
                # while (episode_masks[-1][k] > 0):
                #     step_string = []
                #     regen_strings = obs_preds_last_batch[k].argmax(-1)
                #     for l in range(len(regen_strings)):
                #         step_string.append(agent.word_vocab[regen_strings[l]])
                #     ep_string_obs_tf.append((' '.join(step_string).split("<eos>")[0]))
                #     k += 1
                #     if k == len(episode_masks[-1]):
                #         break

                ep_string_goal_tf = []
                ep_string_obs_tf = []
                for k in range(len(goal_preds_tf_batch)):
                    regen_strings = goal_preds_tf_batch[k].argmax(-1)
                    step_string = []
                    for l in range(len(regen_strings)):
                        step_string.append(agent.word_vocab[regen_strings[l]])
                    ep_string_goal_tf.append((' '.join(step_string).split("<eos>")[0]))

                    regen_strings = obs_preds_last_batch[k].argmax(-1)
                    step_string = []
                    for l in range(len(regen_strings)):
                        step_string.append(agent.word_vocab[regen_strings[l]])
                    ep_string_obs_tf.append((' '.join(step_string).split("<eos>")[0]))

                ep_string_goal_greedy = goal_preds_greedy_words_batch
                ep_string_obs = goal_preds_obs_batch

                if agent.run_eval:
                    with torch.no_grad():
                        valid_precision, valid_recall, valid_acc, valid_soft_dist_avg, log_msg_append = \
                            evaluate_reward_predictor_rnn(env=env,
                                                          agent=agent,
                                                          max_len_force=max_len_force,
                                                          eval_max_counter=eval_max_counter,
                                                          valid_test="test",
                                                          apply_history_reward_loss=apply_history_reward_loss)
                        env.split_reset("train")
                else:
                    valid_precision, valid_recall, valid_acc, valid_soft_dist_avg, log_msg_append = 0, 0, 0, 0, ''

                if valid_precision + valid_recall > 0:
                    valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall)
                else:
                    valid_f1 = 0

                if valid_f1 > best_f1_so_far:
                    best_f1_so_far = valid_f1
                    agent.save_model_to_path(save_to_path,
                                             episode_no=episode_no,
                                             eval_acc=valid_f1,
                                             eval_loss=None,
                                             log_file=log_file,
                                             split_optimizer=split_optimizer)

                ave_valid_reward_precision.push(valid_precision)
                ave_valid_reward_recall.push(valid_recall)
                ave_valid_reward_f1.push(valid_f1)
                ave_valid_reward_acc.push(valid_acc)
                ave_valid_reward_soft_dist.push(valid_soft_dist_avg)

                print(
                    "Episode: {:3d} | time spent: {:s} | Max len {:2.1f} | weight {:2.5f} | learning rate {:2.8f}| "
                    "train loss: {:2.3f} | reward loss: {:2.3f} | latent loss: {:2.3f} | obs loss: {:2.3f} | "
                    "Reward train acc: {:2.3f} | Reward train f1: {:2.3f} | "
                    "Reward train precision: {:2.3f} | Reward train recall: {:2.3f} |"
                    "Reward valid acc: {:2.3f} | Reward valid f1: {:2.3f} | "
                    "Reward valid precision: {:2.3f} | Reward valid recall: {:2.3f} | "
                    "Reward soft dist: {:2.3f} "
                        .format(
                        episode_no, str(time_2 - time_1).rsplit(".")[0], max_len, weight,
                        learning_rate, ave_train_loss.get_avg(), ave_train_reward_loss.get_avg(),
                        ave_train_latent_loss.get_avg(), ave_train_obs_loss.get_avg(),
                        ave_train_reward_acc.get_avg(), ave_train_reward_f1.get_avg(),
                        ave_train_reward_precision.get_avg(), ave_train_reward_recall.get_avg(),
                        valid_acc, valid_f1,
                        valid_precision, valid_recall,
                        valid_soft_dist_avg,
                    ), file=log_file, flush=True)

                print('| Goal loss: {0} | Observation: {1} | '
                      'Predicted Goal Sentences with Teaching Force: {2} and Greedy: {3} | '
                      'Predicted Obs Sentences with Teaching Force: {4}. '.format(
                    ave_train_goal_loss.get_avg(),
                    ep_string_obs,
                    ep_string_goal_tf,
                    ep_string_goal_greedy,
                    ep_string_obs_tf,
                ),
                    file=log_file,
                    flush=True
                )

    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        print('--------------------------------------------', file=log_file, flush=True)
        print('Exiting from training early...', file=log_file, flush=True)
        if log_file is not None:
            log_file.close()


if __name__ == '__main__':
    args = read_args()
    if int(args.TRAIN_FLAG):
        train(args)
    else:
        test(args)
