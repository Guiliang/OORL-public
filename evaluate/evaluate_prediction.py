import copy
import torch
import numpy as np
import tqdm
import torch.nn.functional as F
from generic.data_utils import merge_sample_triplet_index, get_goal_sentence, process_facts, serialize_facts, \
    adj_to_triplets, matching_object_from_obs, diff_triplets, extract_goal_sentence_from_obs, handle_rnn_max_len
from generic.model_utils import to_np, to_pt


def compute_dynamic_predict_acc(agent, graph_diff_pos_sample_mask, graph_diff_neg_sample_mask, graph_negative_mask,
                                input_adj_m, output_adj_m, actions, observations, sample_triplet_index, hx, cx,
                                batch_mask, pos_mask, neg_mask, rewards=None, check_probing_acc=False):
    list_eval_acc, list_eval_loss = [], []
    list_score, list_soft_dist = [], []
    tp_num, fp_num, tn_num, fn_num = 0, 0, 0, 0
    list_real_label = []

    with torch.no_grad():
        pred_loss_t, diff_loss_t, latent_loss_t, pred_adj, real_adj, hx, cx = \
            agent.get_predict_dynamics_logits(input_adj_m=input_adj_m,
                                              actions=actions,
                                              observations=observations,
                                              output_adj_m=output_adj_m,
                                              hx=hx,
                                              cx=cx,
                                              graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                                              graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                                              graph_negative_mask=graph_negative_mask,
                                              if_loss_mean=False,
                                              batch_mask=batch_mask,
                                              pos_mask=pos_mask,
                                              neg_mask=neg_mask
                                              )

    loss = to_np(pred_loss_t) if graph_diff_pos_sample_mask is None else to_np(diff_loss_t)
    rm_num = 0
    abs_diff = np.abs(to_np(pred_adj) - to_np(real_adj))
    abs_diff_mean = np.mean(abs_diff)
    list_soft_dist = [abs_diff_mean]
    for batch_idx in range(len(sample_triplet_index)):
        correct_num = 0
        batch_sample_triplet_index = sample_triplet_index[batch_idx]  # [num_triple, 3]
        if len(batch_sample_triplet_index) == 0:
            rm_num += 1
            continue
        for i in range(len(batch_sample_triplet_index)):
            triplet_index = batch_sample_triplet_index[i]
            score = pred_adj[batch_idx - rm_num][triplet_index[0], triplet_index[1], triplet_index[2]].cpu().item()
            real_label = real_adj[batch_idx - rm_num][triplet_index[0], triplet_index[1], triplet_index[2]].cpu().item()
            list_real_label.append(real_label)

            if score > 0.5:
                predicted_edge_label = 1
                if real_label == predicted_edge_label:
                    correct_num += 1
                    tp_num += 1
                else:
                    fp_num += 1
            else:
                predicted_edge_label = 0
                if real_label == predicted_edge_label:
                    correct_num += 1
                    tn_num += 1
                else:
                    fn_num += 1
            list_score.append(score)
        list_eval_acc.append(float(correct_num) / len(batch_sample_triplet_index))
        list_eval_loss.append(loss[batch_idx - rm_num])
    return list_eval_acc, list_eval_loss, list_score, \
           list_soft_dist, tp_num, fp_num, tn_num, fn_num, hx, cx


def evaluate_graph_prediction(env, agent, log_file, valid_test="valid",
                              eval_max_counter=None,
                              all_triplets_dict={},
                              poss_triplets_mask=None,
                              candidate_triplets_ids=[],
                              filter_mask=None):
    """evaluate the graph prediction performance with the validation dataset"""
    env.split_reset(valid_test)
    agent.eval()
    eval_counter = 0

    diff_list_eval_acc_all, gen_list_eval_acc_all, \
    list_eval_loss_all, list_eval_soft_dist_all = [], [], [], []

    list_score_all, p_pred_all, n_pred_all = [], 0, 0
    hx, cx = None, None
    diff_tp_num_sum, diff_fp_num_sum, diff_tn_num_sum, diff_fn_num_sum = 0, 0, 0, 0
    gen_tp_num_sum, gen_fp_num_sum, gen_tn_num_sum, gen_fn_num_sum = 0, 0.0, 0, 0

    # gen_list_eval_acc, gen_list_eval_loss, gen_list_score = [], [], []
    # gen_p_pred, gen_n_pred = 0, 0
    pbar = None
    if valid_test == 'test':
        if eval_max_counter is not None:
            test_length = eval_max_counter
        else:
            test_length = env.data_size / env.evaluate_batch_size + 1
        pbar = tqdm(total=test_length, desc='Running Testing')
    while True:
        if eval_max_counter is not None and eval_counter >= eval_max_counter:  # for debugging
            break
        target_graph_triplets, previous_graph_triplets, actions, current_observations = env.get_batch()

        if "dynamic_predict_ims" in agent.task:
            # evaluate the difference before and after the action
            input_adjacency_matrix = agent.get_graph_adjacency_matrix(previous_graph_triplets)
            output_adjacency_matrix = agent.get_graph_adjacency_matrix(target_graph_triplets)
            graph_diff_pos_sample_mask, graph_diff_neg_sample_mask, diff_triplet_index, \
            input_adjacency_matrix, output_adjacency_matrix, actions, \
            pos_mask, neg_mask = \
                agent.get_diff_sample_masks(prev_adjs=input_adjacency_matrix,
                                            target_adjs=output_adjacency_matrix,
                                            actions=actions,)

            # evaluate decoding perfomance
            if agent.sample_number == 'None':  # do not sample
                filter_mask_batch = np.repeat(poss_triplets_mask, len(output_adjacency_matrix), axis=0)
                graph_negative_mask_agg = filter_mask_batch - output_adjacency_matrix
                tmp_min = np.min(graph_negative_mask_agg)
                assert tmp_min == 0
                tmp_max = np.max(graph_negative_mask_agg)
                assert tmp_max == 1
                sample_triplet_index_agg = [candidate_triplets_ids] * len(output_adjacency_matrix)
            else:
                graph_negative_mask_list, sample_number, sample_triplet_index_list = \
                    agent.get_graph_negative_sample_mask(adjs=output_adjacency_matrix,
                                                         triplets=target_graph_triplets,
                                                         sample_number=agent.sample_number,
                                                         all_triplets_dict=all_triplets_dict)

                assert len(graph_diff_pos_sample_mask) == len(graph_negative_mask_list[0])
                graph_negative_mask_agg = np.zeros(shape=output_adjacency_matrix.shape)
                for sample_index in range(sample_number):
                    graph_negative_mask = graph_negative_mask_list[sample_index]
                    graph_negative_mask_agg += graph_negative_mask
                sample_triplet_index_agg = merge_sample_triplet_index(sample_triplet_index_list)

            if len(diff_triplet_index) == 0:
                continue

            if 'random' in agent.task:
                input_adjacency_matrix = np.random.uniform(low=0.0, high=1.0, size=input_adjacency_matrix.shape)
            elif 'gtf' in agent.task:
                input_adjacency_matrix = output_adjacency_matrix

            diff_list_eval_acc, diff_list_eval_loss, diff_list_score, diff_list_soft_dist, \
            diff_tp_num, diff_fp_num, diff_tn_num, diff_fn_num, hx, cx = \
                compute_dynamic_predict_acc(agent=agent,
                                            graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                                            graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                                            graph_negative_mask=graph_negative_mask_agg,
                                            input_adj_m=input_adjacency_matrix,
                                            output_adj_m=output_adjacency_matrix,
                                            actions=actions,
                                            observations=current_observations,
                                            sample_triplet_index=diff_triplet_index,
                                            hx=hx,
                                            cx=cx,
                                            pos_mask=pos_mask,
                                            neg_mask=neg_mask,
                                            batch_mask=None, )
            # filter_mask=filter_mask)

            gen_list_eval_acc, gen_list_eval_loss, gen_list_score, gen_list_soft_dist, \
            gen_tp_num, gen_fp_num, gen_tn_num, gen_fn_num, hx, cx = \
                compute_dynamic_predict_acc(agent=agent,
                                            graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                                            graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                                            graph_negative_mask=graph_negative_mask_agg,
                                            input_adj_m=input_adjacency_matrix,
                                            output_adj_m=output_adjacency_matrix,
                                            actions=actions,
                                            observations=current_observations,
                                            sample_triplet_index=sample_triplet_index_agg,
                                            hx=hx,
                                            cx=cx,
                                            pos_mask=pos_mask,
                                            neg_mask=neg_mask,
                                            batch_mask=None, )
            # filter_mask=filter_mask)
        elif "graph_autoenc" in agent.task:
            real_adjacency_matrix = agent.get_graph_adjacency_matrix(target_graph_triplets)
            if 'random' in agent.task:
                input_adjacency_matrix = np.random.uniform(low=0.0, high=1.0, size=real_adjacency_matrix.shape)
            else:
                input_adjacency_matrix = real_adjacency_matrix

            graph_negative_mask_list, sample_number, sample_triplet_index_list = agent.get_graph_negative_sample_mask(
                adjs=real_adjacency_matrix,
                triplets=target_graph_triplets,
                sample_number=10,
                all_triplets_dict=all_triplets_dict,
            )
            graph_negative_mask_agg = np.zeros(shape=real_adjacency_matrix.shape)
            for sample_index in range(sample_number):
                graph_negative_mask = graph_negative_mask_list[sample_index]
                graph_negative_mask_agg += graph_negative_mask

            gen_list_eval_acc, gen_list_eval_loss, gen_list_score, gen_list_soft_dist, \
            gen_tp_num, gen_tn_num, gen_fp_num, gen_fn_num = \
                compute_auto_encoder_acc(agent=agent,
                                         graph_negative_mask=graph_negative_mask_agg,
                                         input_adj_m=input_adjacency_matrix,
                                         real_adj_m=real_adjacency_matrix,
                                         sample_triplet_index=sample_triplet_index_list[0])
            diff_list_eval_acc, diff_list_eval_loss, diff_list_score, diff_list_soft_dist = [], [], [], []
            diff_tp_num, diff_fp_num, diff_tn_num, diff_fn_num = 0, 0, 0, 0
        else:
            raise ValueError("Unknown task type {0}".format(agent.task))
        diff_list_eval_acc_all += diff_list_eval_acc
        gen_list_eval_acc_all += gen_list_eval_acc
        diff_tp_num_sum += diff_tp_num
        diff_fp_num_sum += diff_fp_num
        diff_tn_num_sum += diff_tn_num
        diff_fn_num_sum += diff_fn_num
        gen_tp_num_sum += gen_tp_num
        gen_fp_num_sum += gen_fp_num
        gen_tn_num_sum += gen_tn_num
        gen_fn_num_sum += gen_fn_num
        list_eval_loss_all += diff_list_eval_loss
        list_eval_loss_all += gen_list_eval_loss
        list_score_all += diff_list_score
        list_score_all += gen_list_score
        list_eval_soft_dist_all += diff_list_soft_dist
        list_eval_soft_dist_all += gen_list_soft_dist
        p_pred_all += (diff_tp_num + diff_fp_num)
        p_pred_all += (gen_tp_num + gen_fp_num)
        n_pred_all += (diff_tn_num + diff_fn_num)
        n_pred_all += (gen_tn_num + gen_fn_num)
        eval_counter += 1
        if valid_test == 'test':
            pbar.update(1)
        if env.batch_pointer == 0:
            break

    avg_score = np.mean(list_score_all)
    log_msg_append = " | avg score: {:2.3f} | n_positive: {:3d} | n_negative: {:3d} |\n".format(avg_score,
                                                                                                p_pred_all,
                                                                                                n_pred_all)

    if diff_tp_num_sum + diff_fp_num_sum > 0:
        diff_precision = float(diff_tp_num_sum) / (diff_tp_num_sum + diff_fp_num_sum)
    else:
        diff_precision = 0

    if diff_tp_num_sum + diff_fn_num_sum > 0:
        diff_recall = float(diff_tp_num_sum) / (diff_tp_num_sum + diff_fn_num_sum)
    else:
        diff_recall = 0

    if diff_recall > 0 and diff_precision > 0:
        diff_f1 = 2 * diff_precision * diff_recall / (diff_precision + diff_recall)
    else:
        diff_f1 = 0

    if diff_tp_num_sum + diff_fp_num_sum + diff_tn_num_sum + diff_fn_num_sum == 0:
        diff_acc = 0
    else:
        diff_acc = float(diff_tp_num_sum + diff_tn_num_sum) / (
                diff_tp_num_sum + diff_fp_num_sum + diff_tn_num_sum + diff_fn_num_sum)

    if gen_tp_num_sum + gen_fp_num_sum > 0:
        gen_precision = float(gen_tp_num_sum) / (gen_tp_num_sum + gen_fp_num_sum)
    else:
        gen_precision = 0

    if gen_tp_num_sum + gen_fn_num_sum > 0:
        gen_recall = float(gen_tp_num_sum) / (gen_tp_num_sum + gen_fn_num_sum)
    else:
        gen_recall = 0

    if gen_recall > 0 and gen_precision > 0:
        gen_f1 = 2 * gen_precision * gen_recall / (gen_precision + gen_recall)
    else:
        gen_f1 = 0

    if gen_tp_num_sum + gen_fp_num_sum + gen_tn_num_sum + gen_fn_num_sum == 0:
        gen_acc = 0
    else:
        gen_acc = float(gen_tp_num_sum + gen_tn_num_sum) / (
                gen_tp_num_sum + gen_fp_num_sum + gen_tn_num_sum + gen_fn_num_sum)

    return np.mean(list_eval_loss_all), \
           diff_acc, gen_acc, \
           diff_precision, gen_precision, \
           diff_recall, gen_recall, \
           diff_f1, gen_f1, \
           np.mean(list_eval_soft_dist_all), log_msg_append


def test_on_textworld_supervised(agent_planning, agent_extractor, game_difficulty_level,
                                 filter_mask, input_adj_m, action, obs, hx, cx, _last_facts, infos,
                                 goal_sentences, goal_sentence_store, ingredients, log_file,
                                 action_counter=0,
                                 save_object_prior_embeddings=None, save_object_posterior_embeddings=None,
                                 ):
    observation_string, action_candidate_list = agent_planning.get_game_info_at_certain_step_lite(obs, infos)
    if " your score has just gone up by one point ." in observation_string:
        observation_string = observation_string.replace(" your score has just gone up by one point .", "")
    observations = [observation_string]
    actions = [action]

    _facts_seen = process_facts(_last_facts,
                                infos["game"],
                                infos["facts"],
                                infos["last_action"],
                                action)
    triplets_real = [sorted(serialize_facts(_facts_seen))]

    print(triplets_real[0], file=log_file, flush=True)
    triplets_input = adj_to_triplets(adj_matrix=input_adj_m,
                                     node_vocab=agent_planning.node_vocab,
                                     relation_vocab=agent_planning.relation_vocab)

    predicted_encodings, hx_new, cx_new, \
    attn_mask, input_node_name, input_relation_name, node_encodings, relation_encodings, _, _ = \
        agent_extractor.compute_updated_dynamics(input_adj_m, actions, observations, hx, cx)
    pred_extract_adj = agent_extractor.model.decode_graph(predicted_encodings, relation_encodings)
    filter_mask = np.repeat(filter_mask, len(input_adj_m), axis=0)
    # for i in range(3):
    # plot_heatmap(to_np(pred_extract_adj)[0, 2, :, :])
    adj_matrix = to_np(pred_extract_adj)
    adj_matrix = filter_mask * adj_matrix
    adj_matrix = (adj_matrix > 0.5).astype(int)
    triplets_extraction_pred = adj_to_triplets(adj_matrix=adj_matrix,
                                               node_vocab=agent_extractor.node_vocab,
                                               relation_vocab=agent_extractor.relation_vocab)
    triplets_extraction_pred = matching_object_from_obs(observations=observations,
                                                        actions=actions,
                                                        node_vocab=agent_extractor.node_vocab,
                                                        pred_triplets=triplets_extraction_pred,
                                                        input_triplets=triplets_input,
                                                        diff_level=game_difficulty_level)
    pred_extraction_adj_matrix = agent_extractor.get_graph_adjacency_matrix(triplets_extraction_pred)

    # for i in range(3):
    # plot_heatmap(to_np(pred_extraction_adj_matrix)[0, 2, :, :])
    # print(sorted(triplets_extraction_pred[0]), file=log_file, flush=True)

    if agent_planning.model.dynamic_loss_type == 'latent':
        predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
        predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
        input_node_name, input_relation_name, node_encodings, relation_encodings, \
        action_encodings_sequences, action_mask = \
            agent_planning.compute_updated_dynamics(input_adj_m, actions, observations, hx, cx)
        predicted_encodings = predicted_encodings_prior
    else:
        raise ValueError("This is not done yet.")

    if save_object_prior_embeddings is None:
        save_object_prior_embeddings = to_np(predicted_encodings_prior)
    else:
        save_object_prior_embeddings = np.concatenate([save_object_prior_embeddings,
                                                       to_np(predicted_encodings_prior)],
                                                      axis=0)

    if save_object_posterior_embeddings is None:
        save_object_posterior_embeddings = to_np(predicted_encodings_post)
    else:
        save_object_posterior_embeddings = np.concatenate([save_object_posterior_embeddings,
                                                           to_np(predicted_encodings_post)],
                                                          axis=0)

    pred_planning_adj = agent_planning.model.decode_graph(predicted_encodings, relation_encodings)
    filter_mask = np.repeat(filter_mask, len(input_adj_m), axis=0)
    adj_matrix = to_np(pred_planning_adj)
    adj_matrix = filter_mask * adj_matrix

    adj_matrix = (adj_matrix > 0.5).astype(int)
    triplets_planning_pred = adj_to_triplets(adj_matrix=adj_matrix,
                                             node_vocab=agent_planning.node_vocab,
                                             relation_vocab=agent_planning.relation_vocab)
    triplets_planning_pred = matching_object_from_obs(observations=observations,
                                                      actions=actions,
                                                      node_vocab=agent_planning.node_vocab,
                                                      pred_triplets=triplets_planning_pred,
                                                      input_triplets=triplets_input,
                                                      diff_level=game_difficulty_level)

    print(sorted(triplets_planning_pred[0]), file=log_file, flush=True)
    tmp_diff, num_redundant, num_lack = diff_triplets(triplets1=triplets_planning_pred[0],
                                                      triplets2=triplets_real[0])
    print(tmp_diff, file=log_file, flush=True)
    # sum_diff = np.sum(np.abs(tmp_diff))

    goal_sentence_next, ingredients, goal_sentence_store = \
        get_goal_sentence(pre_goal_sentence=goal_sentences,
                          obs=observation_string,
                          state=pred_extraction_adj_matrix,
                          ingredients=copy.copy(ingredients),
                          node2id=agent_planning.node2id,
                          relation2id=agent_planning.relation2id,
                          node_vocab=agent_planning.node_vocab,
                          relation_vocab=agent_planning.relation_vocab,
                          goal_sentence_store=goal_sentence_store,
                          difficulty_level=game_difficulty_level,
                          recheck_ingredients=False)

    print("Goal: {0}, \nAction: '{1}',\nObservation: '{2}',\n"
          "Diff num: {3}, num_redundant: {4}, num_lack {5}\n".format(goal_sentences,
                                                                     action,
                                                                     observation_string,
                                                                     len(tmp_diff),
                                                                     num_redundant,
                                                                     num_lack),
          file=log_file, flush=True)

    return pred_extraction_adj_matrix, action_candidate_list, _facts_seen, goal_sentence_next, \
           goal_sentence_store, ingredients, save_object_prior_embeddings, save_object_posterior_embeddings


def test_on_textworld_unsupervised(agent_planning, game_difficulty_level,
                                   input_adj_m, action, obs, goal_sentences, ingredients, goal_sentence_store,
                                   hx, cx, _last_facts, infos,
                                   save_object_prior_embeddings=None,
                                   save_object_posterior_embeddings=None,
                                   action_counter=0
                                   ):
    observation_string, action_candidate_list = agent_planning.get_game_info_at_certain_step_lite(obs, infos)

    if " your score has just gone up by one point ." in observation_string:
        observation_string = observation_string.replace(" your score has just gone up by one point .", "")
    observations = [observation_string]
    actions = [action]

    _facts_seen = process_facts(_last_facts,
                                infos["game"],
                                infos["facts"],
                                infos["last_action"],
                                action)

    predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
    predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
    input_node_name, input_relation_name, node_encodings, relation_encodings, \
    _, _ = \
        agent_planning.compute_updated_dynamics(input_adj_m=input_adj_m,
                                                actions=actions,
                                                observations=observations,
                                                goal_sentences=None,
                                                rewards=to_pt(np.asarray([0]), enable_cuda=agent_planning.use_cuda,
                                                              type='float'),
                                                hx=hx,
                                                cx=cx, )
    if save_object_prior_embeddings is None:
        save_object_prior_embeddings = to_np(predicted_encodings_prior)
    else:
        save_object_prior_embeddings = np.concatenate([save_object_prior_embeddings,
                                                       to_np(predicted_encodings_prior)],
                                                      axis=0)

    if save_object_posterior_embeddings is None:
        save_object_posterior_embeddings = to_np(predicted_encodings_post)
    else:
        save_object_posterior_embeddings = np.concatenate([save_object_posterior_embeddings,
                                                           to_np(predicted_encodings_post)],
                                                          axis=0)

    goal_sentences_next, ingredients, goal_sentence_store = \
        extract_goal_sentence_from_obs(
            agent=agent_planning,
            object_encodings=predicted_encodings_post,
            object_mask=node_mask,
            pre_goal_sentence=goal_sentences,
            obs=observation_string,
            obs_origin=obs,
            ingredients=copy.copy(ingredients),
            goal_sentence_store=goal_sentence_store,
            difficulty_level=game_difficulty_level,
            rule_based_extraction=False)

    predicted_encodings = predicted_encodings_post
    predict_output_adj = agent_planning.model.decode_graph(predicted_encodings, relation_encodings)
    input_actions = agent_planning.get_word_input(actions, minimum_len=10)  # batch x action_len
    action_encodings_sequences, action_mask = \
        agent_planning.model.encode_text_for_reward_prediction(input_actions)

    print("Action: '{0}', \nObservation: '{1}'".format(action, observation_string))
    for goal_sentence in goal_sentences:
        input_goals = agent_planning.get_word_input([goal_sentence], minimum_len=20)  # batch x goal_len
        goal_encodings_sequences, goal_mask = \
            agent_planning.model.encode_text_for_reward_prediction(input_goals)

        pred_rewards = agent_planning.compute_rewards_unsupervised(
            predicted_encodings=predicted_encodings,
            node_mask=node_mask,
            action_encodings_sequences=action_encodings_sequences,
            action_mask=action_mask,
            goal_encodings_sequences=goal_encodings_sequences,
            goal_mask=goal_mask)
        pred_rewards = F.softmax(pred_rewards, dim=1)
        pred_rewards_label = 1 - torch.argmax(pred_rewards).item()
        print("Goal: [{0}] with reward: {1} and confidence: {2}.".format(goal_sentence,
                                                                         pred_rewards_label,
                                                                         pred_rewards[0][
                                                                             1 - pred_rewards_label].item()))
    print('')

    return predict_output_adj, action_candidate_list, _facts_seen, \
           goal_sentences_next, ingredients, goal_sentence_store, \
           save_object_prior_embeddings, save_object_posterior_embeddings


def evaluate_reward_predictor_rnn(env, agent, max_len_force, eval_max_counter,
                                  valid_test="valid", apply_history_reward_loss=True, add_goal=True):
    env.split_reset(valid_test)
    agent.eval()
    true_positive_num, true_negative_num, false_positive_num, false_negative_num = 0, 0, 0, 0
    soft_dist_all = []
    eval_loss_all = []
    counter = 0
    to_print = []

    while (True):
        if eval_max_counter is not None and counter >= eval_max_counter:  # for debugging
            break
        current_triplets, previous_triplets, current_observations, previous_actions, \
        current_rewards, current_goal_sentences = env.get_batch()
        curr_batch_size = len(current_triplets)

        origin_lens = [len(elem) for elem in current_observations]
        max_len = max(origin_lens) if max_len_force is None else max_len_force
        curr_batch_size = len(current_observations)

        input_dense_adj_t = np.zeros(
            (curr_batch_size, len(agent.relation_vocab), len(agent.node_vocab), len(agent.node_vocab)),
            dtype="float32")

        current_triplets, lens = handle_rnn_max_len(max_len, current_triplets, [])
        previous_triplets, lens = handle_rnn_max_len(max_len, previous_triplets, [])
        current_observations, lens = handle_rnn_max_len(max_len, current_observations, "<pad>")
        current_goal_sentences, lens = handle_rnn_max_len(max_len, current_goal_sentences, "<pad>")
        current_rewards, lens = handle_rnn_max_len(max_len, current_rewards, 0)
        previous_actions, lens = handle_rnn_max_len(max_len, previous_actions, "<pad>")
        masks = torch.zeros((curr_batch_size, max_len),
                            dtype=torch.float).cuda() if agent.use_cuda else torch.zeros((curr_batch_size, max_len),
                                                                                         dtype=torch.float)

        if apply_history_reward_loss:
            for i in range(curr_batch_size):
                masks[i, :lens[i]] = 1
        else:
            for i in range(curr_batch_size):
                masks[i, lens[i] - 1] = 1
        counter += 1
        prev_h = None
        for i in range(max_len):
            current_t_triplets = [elem[i] for elem in current_triplets]
            previous_t_triplets = [elem[i] for elem in previous_triplets]
            current_t_observations = [elem[i] for elem in current_observations]
            current_t_goal_sentences = [elem[i] for elem in current_goal_sentences]
            current_t_rewards = [elem[i] for elem in current_rewards]
            current_t_adjacency_matrix = agent.get_graph_adjacency_matrix(current_t_triplets)
            previous_t_adjacency_matrix = agent.get_graph_adjacency_matrix(previous_t_triplets)
            previous_t_actions = [elem[i] for elem in previous_actions]
            if agent.task == 'reward_prediction':
                loss, pred_rewards, real_rewards, correct_count, rnn_output = \
                    agent.reward_prediction_supervised(current_adjacency_matrix=current_t_adjacency_matrix,
                                                       real_rewards=current_t_rewards,
                                                       current_observations=current_t_observations,
                                                       previous_actions=previous_t_actions,
                                                       goal_sentences=current_t_goal_sentences,
                                                       if_apply_rnn=agent.model.reward_predictor_apply_rnn,
                                                       h_t_minus_one=prev_h,
                                                       episode_masks=masks[:, i],
                                                       if_loss_mean=True)
                prev_h = rnn_output
            elif agent.task == "dynamic_reward_prediction":
                loss, pred_rewards, real_rewards, correct_count, rnn_output = \
                    agent.reward_prediction_dynamic(previous_adjacency_matrix=previous_t_adjacency_matrix,
                                                    real_rewards=current_t_rewards,
                                                    current_observations=current_t_observations,
                                                    previous_actions=previous_t_actions,
                                                    current_goal_sentences=current_t_goal_sentences,
                                                    if_apply_rnn=agent.model.reward_predictor_apply_rnn,
                                                    h_t_minus_one=prev_h,
                                                    episode_masks=masks[:, i],
                                                    if_loss_mean=True)
                prev_h = rnn_output
            elif 'unsupervised' in agent.task:
                if not add_goal:
                    current_t_goal_sentences = None
                obs_loss_t, predict_adj_t, obs_pred_t, \
                loss, pred_rewards, real_rewards, latent_loss_t, _, _, _ = \
                    agent.get_unsupervised_dynamics_logistic(
                        input_adj_m=input_dense_adj_t,
                        actions=previous_t_actions,
                        observations=current_t_observations,
                        real_rewards=current_t_rewards,
                        goal_sentences=current_t_goal_sentences,
                        batch_masks=masks[:, i],
                        if_loss_mean=True,
                        decode_mode='recon'
                    )

                input_dense_adj_t = predict_adj_t
            else:
                raise ValueError("Unknown reward task name {0}".format(agent.task))

            for bid in range(len(current_triplets)):
                if masks[bid, i]:
                    pred_ = 1 - np.argmax(pred_rewards[bid])  # idx 0 is positive, idx 1 is negative
                    real_ = 1 - np.argmax(real_rewards[bid])  # idx 0 is positive, idx 1 is negative

                    if pred_ == 1 and real_ == 1:
                        true_positive_num += 1
                    elif pred_ == 1 and real_ == 0:
                        false_positive_num += 1
                    elif pred_ == 0 and real_ == 1:
                        false_negative_num += 1
                    elif pred_ == 0 and real_ == 0:
                        true_negative_num += 1

                    soft_dist_by_item = np.absolute(pred_rewards[bid] - real_rewards[bid])
                    soft_dist_all.append(np.sum(soft_dist_by_item))
                    eval_loss_all.append(to_np(loss))
        if env.batch_pointer == 0 and env.type_idx_pointer == -1:
            break

    log_msg_append = " | n_positive: {:3d} | n_negative: {:3d} |".format(true_positive_num + false_positive_num,
                                                                         true_negative_num + false_negative_num)
    if true_positive_num + false_positive_num > 0:
        precision = float(true_positive_num) / (true_positive_num + false_positive_num)
    else:
        precision = 0
    if true_positive_num + false_negative_num > 0:
        recall = float(true_positive_num) / (true_positive_num + false_negative_num)
    else:
        recall = 0
    acc = float(true_positive_num + true_negative_num) / (true_positive_num + true_negative_num + false_positive_num +
                                                          false_negative_num)
    soft_dist_avg = np.mean(np.asarray(soft_dist_all))

    return precision, recall, acc, soft_dist_avg, log_msg_append


def compute_auto_encoder_acc(agent, graph_negative_mask, input_adj_m, real_adj_m, sample_triplet_index):
    list_eval_acc, list_eval_loss = [], []
    list_score, list_soft_dist, p_pred, n_pred = [], [], 0, 0
    list_real_label = []
    tp_num, tn_num, fp_num, fn_num = 0, 0, 0, 0

    with torch.no_grad():
        loss, pred_adj, real_adj = \
            agent.get_graph_autoencoder_logits(input_adj_m=input_adj_m,
                                               real_adj_m=real_adj_m,
                                               graph_negative_mask=graph_negative_mask,
                                               if_loss_mean=False)

    # tmp = np.sum(loss.cpu().data.numpy()) / np.sum(graph_loss_mask)
    # predict_adjacency_matrix = torch.flatten(predict_adjacency_matrix, start_dim=1)
    # real_adjacency_matrix = torch.flatten(adjacency_matrix, start_dim=1)
    loss = to_np(loss)
    for batch_idx in range(len(sample_triplet_index)):
        batch_sample_triplet_index = sample_triplet_index[batch_idx]  # [num_triple, 3]
        correct_num = 0
        total_num = 0
        for i in range(len(batch_sample_triplet_index)):
            triplet_index = batch_sample_triplet_index[i]
            # tmp = predict_adjacency_matrix[batch_idx][triplet_index[0], triplet_index[1], triplet_index[2]]
            # print(tmp)
            total_num += 1
            score = pred_adj[batch_idx][triplet_index[0], triplet_index[1], triplet_index[2]].cpu().item()
            real_label = real_adj[batch_idx][triplet_index[0], triplet_index[1], triplet_index[2]].cpu().item()
            list_real_label.append(real_label)
            soft_dist = (score - real_label) ** 2
            if score > 0.5:
                predicted_edge_label = 1
                p_pred += 1
                if real_label == predicted_edge_label:
                    tp_num += 1
                    correct_num += 1
                else:
                    fp_num += 1
            else:
                predicted_edge_label = 0
                n_pred += 1
                if real_label == predicted_edge_label:
                    tn_num += 1
                    correct_num += 1
                else:
                    fn_num += 1
            list_score.append(score)
            list_soft_dist.append(soft_dist)

        list_eval_loss.append(loss[batch_idx])
        list_eval_acc.append(float(correct_num) / total_num)

    return list_eval_acc, list_eval_loss, \
           list_score, list_soft_dist, tp_num, tn_num, fp_num, fn_num