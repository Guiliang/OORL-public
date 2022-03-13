import copy
import datetime
import os

import numpy as np
import torch
import yaml

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)

from generic.reward_prediction_dataset import RewardPredictionDynamicDataGoal
from agent.agent import OORLAgent
from evaluate.evaluate_prediction import evaluate_reward_predictor_goal
from generic.data_utils import load_config, generate_triplets_filter_mask, \
    adj_to_triplets, matching_object_from_obs, get_goal_sentence, read_args, diff_triplets
from generic.model_utils import HistoryScoreCache, to_np, load_graph_extractor


def test(args):
    import textworld
    from generic.data_utils import process_facts, serialize_facts
    config, debug_mode, log_file_path, model_name = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

    config['general']['checkpoint']['load_pretrained'] = False
    agent_planning = OORLAgent(config, log_file, '')
    if agent_planning.difficulty_level == 'mixed':
        game_difficulty_level = 3
    else:
        game_difficulty_level = agent_planning.difficulty_level
    if agent_planning.difficulty_level == 'mixed':
        test_path = '../source/dataset/rl.0.2/test/difficulty_level_{0}/'.format(game_difficulty_level)
    else:
        test_path = '../source/dataset/rl.0.2/test/difficulty_level_{0}/'.format(agent_planning.difficulty_level)

    if agent_planning.difficulty_level == 'mixed':
        candidate_triplets = []
        for difficulty_level in [3, 5, 7, 9]:
            set_dir = config["graph_auto"]["data_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
                difficulty_level)
            with open(set_dir, 'r') as f:
                candidate_triplets_sub = f.readlines()
            candidate_triplets += candidate_triplets_sub
    else:
        set_dir = config["graph_auto"]["data_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
            agent_planning.difficulty_level)
        with open(set_dir, 'r') as f:
            candidate_triplets = f.readlines()

    filter_mask = generate_triplets_filter_mask(triplet_set=candidate_triplets,
                                                node2id=agent_planning.node2id,
                                                relation2id=agent_planning.relation2id)

    load_path = agent_planning.output_dir + agent_planning.experiment_tag + "/difficulty_level_{3}/" \
                                                                            "saved_model_Dynamic_" \
                                                                            "Reward_Predictor_Goal_Linear_{2}.pt".format(
        agent_planning.model.dynamic_model_mechanism,
        agent_planning.model.dynamic_loss_type,
        'Aug-31-2021', # 'Aug-15-2021_real_goal',  # Jun-13-2021_real_goal, Apr-27-2021_real_goal, May-15-2021
        agent_planning.difficulty_level)

    agent_planning.load_pretrained_model(load_path, log_file=log_file, load_partial_graph=False)

    print("\n\n" + "*" * 30 + "Start Loading extractor" + "*" * 30, file=log_file, flush=True)
    if agent_planning.difficulty_level == 'mixed':
        extractor_config_dir = '../configs/predict_graphs_dynamics_linear_seen_fineTune_df-mixed.yaml'
    else:
        extractor_config_dir = '../configs/predict_graphs_dynamics_linear_seen_fineTune_df{0}.yaml'.format(
                agent_planning.difficulty_level)

    with open(extractor_config_dir) as reader:
        extract_config = yaml.safe_load(reader)
    agent_extractor = OORLAgent(extract_config, log_file, '')
    load_graph_extractor(agent_extractor, log_file, difficulty_level=game_difficulty_level)
    print("*" * 30 + "Finish Loading extractor" + "*" * 30, file=log_file, flush=True)

    for test_file in os.listdir(test_path):
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

        def _test_on_textworld(filter_mask, input_adj_m, action, obs, hx, cx, _last_facts, infos,
                               goal_sentences, goal_sentence_store, ingredients):

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
            real_adj_matrix = agent_planning.get_graph_adjacency_matrix([sorted(serialize_facts(_facts_seen))])
            print("Real triplets: {0}".format(triplets_real[0]), file=log_file, flush=True)

            triplets_input = adj_to_triplets(adj_matrix=input_adj_m,
                                             node_vocab=agent_planning.node_vocab,
                                             relation_vocab=agent_planning.relation_vocab)

            predicted_encodings, hx_new, cx_new, \
            attn_mask, input_node_name, input_relation_name, node_encodings, relation_encodings, _, _ = \
                agent_extractor.compute_updated_dynamics(input_adj_m, actions, observations, hx, cx)
            pred_extract_adj = agent_extractor.model.decode_graph(predicted_encodings, relation_encodings)
            filter_mask = np.repeat(filter_mask, len(input_adj_m), axis=0)
            adj_matrix = (to_np(pred_extract_adj) > 0.5).astype(int)
            adj_matrix = filter_mask * adj_matrix
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
            print("Extracted triplets: {0}".format(sorted(triplets_extraction_pred[0])), file=log_file, flush=True)

            predicted_encodings_prior, hx_new_prior, cx_new_prior, \
            predicted_encodings_post, hx_new_post, cx_new_prost, \
            attn_mask, input_node_name, input_relation_name, node_encodings, relation_encodings, _, _ = \
                agent_planning.compute_updated_dynamics(input_adj_m, actions, observations, hx, cx)
            # predicted_encodings = predicted_encodings_prior
            pred_planning_adj = agent_planning.model.decode_graph(predicted_encodings_post, relation_encodings)
            filter_mask = np.repeat(filter_mask, len(input_adj_m), axis=0)
            adj_matrix = (to_np(pred_planning_adj) > 0.5).astype(int)
            pred_planning_adj_matrix = filter_mask * adj_matrix
            triplets_planning_pred = adj_to_triplets(adj_matrix=pred_planning_adj_matrix,
                                                     node_vocab=agent_planning.node_vocab,
                                                     relation_vocab=agent_planning.relation_vocab)
            triplets_planning_pred = matching_object_from_obs(observations=observations,
                                                              actions=actions,
                                                              node_vocab=agent_extractor.node_vocab,
                                                              pred_triplets=triplets_planning_pred,
                                                              input_triplets=triplets_input,
                                                              diff_level=game_difficulty_level)
            # triplets_pred = adj_to_triplets(adj_matrix=pred_adj_matrix,
            #                                 node_vocab=agent.node_vocab,
            #                                 relation_vocab=agent.relation_vocab)
            print("Planning triplets: {0}".format(sorted(triplets_planning_pred[0])), file=log_file, flush=True)
            pred_planning_adj_matrix = agent_extractor.get_graph_adjacency_matrix(triplets_planning_pred)

            tmp_diff, num_redundant, num_lack = diff_triplets(triplets1=triplets_planning_pred[0],
                                                              triplets2=triplets_real[0])
            # print(tmp_diff, file=log_file, flush=True)

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

            print("Ingredients: {0}".format(ingredients))
            for goal_sentence in goal_sentences:
                pred_rewards, rnn_output = agent_planning.compute_rewards(node_encodings=predicted_encodings_prior,
                                                                          node_mask=attn_mask,
                                                                          previous_actions=actions,
                                                                          goal_sentences=[goal_sentence],
                                                                          if_apply_rnn=agent_planning.model.reward_predictor_apply_rnn,
                                                                          h_t_minus_one=None,
                                                                          )
                if agent_planning.use_cuda:
                    rewards_goal = pred_rewards.detach().cpu().numpy()
                else:
                    rewards_goal = pred_rewards.detach().numpy()
                rewards_goal = 1 - np.argmax(rewards_goal[0])  # [pos_prob, neg_prob]
                print("{0}: {1}".format(goal_sentence, rewards_goal))
            print("action: '{0}', \nobservation: '{1}', \ndiff {2}\n".format(action,
                                                                             observation_string,
                                                                             len(tmp_diff)))

            return pred_extraction_adj_matrix, action_candidate_list, _facts_seen, \
                   goal_sentence_next, goal_sentence_store, ingredients

        _last_facts = set()
        obs, infos = env.reset()
        walkthrough = infos["extra.walkthrough"]
        input_adj_m = np.zeros((1,
                                len(agent_planning.relation_vocab),
                                len(agent_planning.node_vocab),
                                len(agent_planning.node_vocab)),
                               dtype="float32")
        action = 'restart'
        hx = None
        cx = None
        goal_sentences, ingredients, goal_sentence_store = \
            get_goal_sentence(pre_goal_sentence=None,
                              ingredients=set(),
                              difficulty_level=game_difficulty_level)

        # while not done:
        for i in range(len(walkthrough) + 1):
            # for i in range(50):
            input_adj_m, action_candidate_list, _last_facts, \
            goal_sentences, goal_sentence_store, ingredients = _test_on_textworld(
                filter_mask,
                input_adj_m,
                action,
                obs,
                hx, cx,
                _last_facts,
                infos,
                goal_sentences,
                goal_sentence_store,
                ingredients)
            # goal_sentences = ['ingredients : carrot', 'ingredients : red hot pepper', 'ingredients : white onion']
            if i < len(walkthrough):
                # if i < 1:
                action = walkthrough[i]
            else:
                action_idx = np.random.choice(len(action_candidate_list))
                action = action_candidate_list[action_idx]
            obs, scores, done, infos = env.step(action)
            # if i == len(walkthrough) - 1:
            #     goal_sentence = 'eat the meal'


def train(args):
    today = datetime.date.today()
    time_1 = datetime.datetime.now()
    config, debug_mode, log_file_path = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    if 'gtf' in config['general']['task']:
        debug_msg = '_gtf_probing'
    elif 'random' in config['general']['task']:
        debug_msg = '_random_probing'
    else:
        debug_msg = ''
    max_data_size = None
    eval_max_counter = None
    if debug_mode:
        config['general']['training']['batch_size'] = 16
        config['general']['checkpoint']['report_frequency'] = 1
        debug_msg += '_debug'
        max_data_size = None
        eval_max_counter = 2
        config['general']['training']['optimizer']['learning_rate'] = 0.0001
        # config['general']['training']['sample_number'] = 1
    print(" Debug mode is {0}".format(debug_mode), file=log_file, flush=True)
    agent = OORLAgent(config, log_file, debug_msg, seed=args.SEED)
    # agent.zero_noise()
    ave_train_loss = HistoryScoreCache(capacity=500)

    if agent.difficulty_level == 'mixed':
        candidate_triplets = []
        for difficulty_level in [3, 5, 7, 9]:
            set_dir = config["graph_auto"]["data_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
                difficulty_level)
            with open(set_dir, 'r') as f:
                candidate_triplets_sub = f.readlines()
            candidate_triplets += candidate_triplets_sub
    else:
        set_dir = config["graph_auto"]["data_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
            agent.difficulty_level)
        with open(set_dir, 'r') as f:
            candidate_triplets = f.readlines()
    filter_mask = generate_triplets_filter_mask(triplet_set=candidate_triplets,
                                                node2id=agent.node2id,
                                                relation2id=agent.relation2id)

    env = RewardPredictionDynamicDataGoal(config, agent, filter_mask, max_data_size, log_file, seed=args.SEED)
    if env.apply_real_goal:
        goal_msg = '_real_goal'
    else:
        goal_msg = ''
    env.split_reset("train")
    if args.FIX_POINT is None:
        save_date_str = today.strftime("%b-%d-%Y")
    else:
        save_date_str = args.FIX_POINT
    save_to_path = agent.output_dir + agent.experiment_tag + "/difficulty_level_{0}/saved_model_{1}_{2}{3}{4}.pt".format(
        env.difficulty_level,
        "Dynamic{0}_Reward_Predictor_Goal_Linear".format('_'+agent.model.dynamic_model_mechanism if agent.model.dynamic_model_mechanism != 'all-independent' else ''),
        save_date_str,
        goal_msg,
        debug_msg)

    episode_no = 0
    best_f1_so_far = 0
    if args.FIX_POINT is not None:
        load_keys, episode_no, loss, best_f1_so_far = agent.load_pretrained_model(load_from=save_to_path, log_file=log_file)

    try:
        while (True):
            if episode_no > agent.max_episode:
                break

            agent.train()
            current_triplets, previous_triplets, current_observations, previous_actions, current_rewards, \
            current_goal_sentences = env.get_batch()

            curr_batch_size = len(current_observations)
            episode_no += curr_batch_size
            train_true_positive_num, train_true_negative_num, \
            train_false_positive_num, train_false_negative_num = 0, 0, 0, 0
            previous_adjacency_matrix = agent.get_graph_adjacency_matrix(previous_triplets)
            current_adjacency_matrix = agent.get_graph_adjacency_matrix(current_triplets)

            loss, pred_rewards, real_rewards, correct_count, _ = \
                agent.reward_prediction_dynamic(previous_adjacency_matrix=previous_adjacency_matrix,
                                                real_rewards=current_rewards,
                                                current_observations=current_observations,
                                                previous_actions=previous_actions,
                                                current_goal_sentences=current_goal_sentences,
                                                if_apply_rnn=agent.model.reward_predictor_apply_rnn,
                                                current_adjacency_matrix=current_adjacency_matrix,
                                                h_t_minus_one=None,
                                                episode_masks=None,
                                                if_loss_mean=True)

            if episode_no % agent.report_frequency <= (
                    episode_no - curr_batch_size) % agent.report_frequency or debug_mode and agent.run_eval:
                for idx in range(len(current_triplets)):
                    pred_ = 1 - np.argmax(pred_rewards[idx])  # idx 0 is positive, idx 1 is negative
                    real_ = 1 - np.argmax(real_rewards[idx])  # idx 0 is positive, idx 1 is negative

                    if pred_ == 1 and real_ == 1:
                        train_true_positive_num += 1
                    elif pred_ == 1 and real_ == 0:
                        train_false_positive_num += 1
                    elif pred_ == 0 and real_ == 1:
                        train_false_negative_num += 1
                    elif pred_ == 0 and real_ == 0:
                        train_true_negative_num += 1
                    else:
                        raise ("ROC Mistake")

            agent.model.zero_grad()
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            loss = to_np(loss)
            ave_train_loss.push(loss)
            # batch_no += 1

            if debug_mode:
                parameters_info = []
                for k, v in agent.model.named_parameters():
                    if v.grad is not None:
                        parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                    else:
                        parameters_info.append("{0}:{1}".format(k, v.grad))
                print(parameters_info, file=log_file, flush=True)

            if episode_no % agent.report_frequency <= (
                    episode_no - curr_batch_size) % agent.report_frequency or debug_mode and agent.run_eval:
                if train_true_positive_num + train_false_positive_num > 0:
                    train_precision = float(train_true_positive_num) / (
                            train_true_positive_num + train_false_positive_num)
                else:
                    train_precision = 0
                if train_true_positive_num + train_false_negative_num > 0:
                    train_recall = float(train_true_positive_num) / (train_true_positive_num + train_false_negative_num)
                else:
                    train_recall = 0
                if train_true_positive_num + train_true_negative_num + train_false_positive_num + train_false_negative_num == 0:
                    raise ("ROC Mistake")

                train_acc = float(train_true_positive_num + train_true_negative_num) / (
                        train_true_positive_num + train_true_negative_num + train_false_positive_num +
                        train_false_negative_num)
                if train_recall + train_precision > 0:
                    train_f1 = 2 * train_precision * train_recall / (train_recall + train_precision)
                else:
                    train_f1 = 0
                with torch.no_grad():
                    valid_precision, valid_recall, valid_acc, valid_soft_dist_avg, log_msg_append = \
                        evaluate_reward_predictor_goal(env, agent, eval_max_counter, "valid")
                    env.split_reset("train")
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
                                             log_file=log_file)

                time_2 = datetime.datetime.now()
                # print(
                #     "Episode: {:3d} | time spent: {:s} | loss: {:2.3f} | valid exact f1: {:2.3f} | valid soft f1: {:2.3f}".format(
                #         episode_no, str(time_2 - time_1).rsplit(".")[0], loss, eval_f1_exact, eval_f1_soft))
                print(
                    "Episode: {:3d} | time spent: {:s} | train loss: {:2.3f} | "
                    "train acc: {:2.3f} | train f1: {:2.3f} | train precision: {:2.3f} | train recall: {:2.3f} |"
                    "valid acc: {:2.3f} | valid f1: {:2.3f} valid precision: {:2.3f} | valid recall: {:2.3f} | valid soft dist: {:2.3f}".format(
                        episode_no, str(time_2 - time_1).rsplit(".")[0], loss,
                        train_acc, train_f1, train_precision, train_recall,
                        valid_acc, valid_f1, valid_precision, valid_recall, valid_soft_dist_avg) + log_msg_append,
                    file=log_file, flush=True)


    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        print('--------------------------------------------')
        print('Exiting from training early...')
    # if agent.run_eval:
    #     if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
    #         print('Evaluating on test set and saving log...')
    #         agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
    #     _, _ = evaluate_pretrained_reward_predictor(env, agent, "test", verbose=True)


if __name__ == '__main__':
    args = read_args()
    if int(args.TRAIN_FLAG):
        train(args)
    else:
        test(args)
