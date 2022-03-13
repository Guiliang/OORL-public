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
from evaluate.evaluate_prediction import evaluate_reward_predictor_rnn
from generic.reward_prediction_dataset import RewardPredictionDynamicDataRNN
from agent.agent import OORLAgent
from generic.data_utils import load_config, read_args, handle_rnn_max_len
from generic.model_utils import HistoryScoreCache, to_np


def train(args):
    time_1 = datetime.datetime.now()
    today = datetime.date.today()
    config, debug_mode, log_file_path = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

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
    train(args)

