import datetime
import os
import random
import math
import torch
import numpy as np
import yaml

from evaluate.evaluate_prediction import compute_dynamic_predict_acc, evaluate_graph_prediction, \
    test_on_textworld_supervised
from generic.radam import HistoryScoreCache

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)

from generic.state_prediction_dataset import SPData
from agent.agent import OORLAgent
from generic.data_utils import _word_to_id, load_config, load_all_possible_set, merge_sample_triplet_index, read_args, \
    generate_triplets_filter_mask, get_goal_sentence
from generic.model_utils import to_np, load_graph_extractor


def test(args):
    import textworld
    config, debug_mode, log_file_path, model_name = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

    agent_planning = OORLAgent(config, log_file, '')

    set_dir = config["graph_auto"]["data_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
        agent_planning.difficulty_level)
    with open(set_dir, 'r') as f:
        candidate_triplets = f.readlines()

    filter_mask = generate_triplets_filter_mask(triplet_set=candidate_triplets,
                                                node2id=agent_planning.node2id,
                                                relation2id=agent_planning.relation2id)

    planning_load_path = agent_planning.output_dir + agent_planning.experiment_tag + "/saved_model_dynamic_{3}_{0}_{4}_{5}_{2}.pt".format(
        agent_planning.model.graph_decoding_method,
        agent_planning.graph_type,
        'df-3_sample-100_weight-1_May-10-2021',
        agent_planning.model.dynamic_model_type,
        agent_planning.model.dynamic_model_mechanism,
        agent_planning.model.dynamic_loss_type,
    )
    agent_planning.load_pretrained_model(planning_load_path,
                                         log_file=log_file,
                                         load_partial_graph=False)
    if agent_planning.model.dynamic_loss_type == 'latent':
        print("\n\n" + "*" * 30 + "Start Loading extractor" + "*" * 30, file=log_file, flush=True)
        with open('../configs/predict_graphs_dynamics_linear_seen_fineTune_df{0}.yaml'.format(
                agent_planning.difficulty_level)) as reader:
            extract_config = yaml.safe_load(reader)
        agent_extractor = OORLAgent(extract_config, log_file, '')
        load_graph_extractor(agent_extractor, log_file)
        print("*" * 30 + "Finish Loading extractor" + "*" * 30, file=log_file, flush=True)
    else:
        agent_extractor = agent_planning

    test_path = '../source/dataset/rl.0.2/test/difficulty_level_{0}/'.format(agent_planning.difficulty_level)
    test_files = []
    for file in os.listdir(test_path):
        if file.endswith('.z8'):
            test_files.append(file)
    test_files = [test_files[1]]
    for test_file in test_files:
        game_file = test_path + test_file
        agent_planning.use_negative_reward = False
        eval_requested_infos = agent_planning.select_additional_infos_lite()
        eval_requested_infos.extras = ["walkthrough"]
        env = textworld.start(game_file, infos=eval_requested_infos)
        env = textworld.envs.wrappers.Filter(env)

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
        done = False
        action_counter = 0
        goal_sentence, ingredients, goal_sentence_store = \
            get_goal_sentence(pre_goal_sentence=None, ingredients=set(),
                              difficulty_level=agent_planning.difficulty_level)

        for i in range(len(walkthrough) + 1):
            # for i in range(50):
            # while True:
            input_adj_m, action_candidate_list, _last_facts, \
            goal_sentence, goal_sentence_store, ingredients, \
            _, _ = \
                test_on_textworld_supervised(agent_planning=agent_planning,
                                             agent_extractor=agent_extractor,
                                             game_difficulty_level=agent_planning.difficulty_level,
                                             filter_mask=filter_mask,
                                             input_adj_m=input_adj_m,
                                             action=action,
                                             obs=obs,
                                             hx=hx,
                                             cx=cx,
                                             _last_facts=_last_facts,
                                             infos=infos,
                                             goal_sentences=goal_sentence,
                                             goal_sentence_store=goal_sentence_store,
                                             ingredients=ingredients,
                                             log_file=log_file,
                                             action_counter=action_counter,
                                             )
            if i < len(walkthrough):
                action = walkthrough[action_counter]
                obs, scores, done, infos = env.step(action)
                action_counter += 1
        break


def train(args):
    random.seed(args.SEED)
    time_1 = datetime.datetime.now()
    today = datetime.date.today()
    config, debug_mode, log_file_path = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

    debug_msg = ''
    eval_max_counter = None
    if debug_mode:
        debug_msg += '_debug'
        config['general']['training']['batch_size'] = 16
        config['general']['checkpoint']['report_frequency'] = 1
        config['general']['training']['optimizer']['learning_rate'] = 0.0005
        config['general']['training']['optimizer']['learning_rate_warmup_until'] = 1000
        eval_max_counter = 2
    print(" Debug mode is {0}".format(debug_mode), file=log_file, flush=True)
    env = SPData(config, args.SEED, log_file)
    env.split_reset("train")
    agent = OORLAgent(config=config,
                      log_file=log_file,
                      debug_msg=debug_msg,
                      seed=args.SEED)
    ave_train_loss = HistoryScoreCache(capacity=500)

    episode_no = 0
    batch_no = 0
    weight_init = 0
    if 'latent' in agent.model.dynamic_loss_type:
        weight_upper_bound = 1
    else:
        weight_upper_bound = weight_init
    hx, cx = None, None
    learning_rate = config['general']['training']['optimizer']['learning_rate']

    if args.FIX_POINT is None:
        save_date_str = today.strftime("%b-%d-%Y")
    else:
        save_date_str = args.FIX_POINT
    save_to_path = agent.output_dir + agent.experiment_tag + "/saved_model_dynamic_linear_{0}_{3}_{4}_df-{5}_sample-{6}_weight-{7}{8}_{1}{2}.pt".format(
        agent.model.graph_decoding_method,
        save_date_str,
        debug_msg,
        agent.model.dynamic_model_mechanism,
        agent.model.dynamic_loss_type,
        env.difficulty_level,
        agent.sample_number,
        weight_upper_bound,
        '-seed-{0}'.format(args.SEED)
    )
    print("The model saving path is: {0}".format(save_to_path), file=log_file, flush=True)

    if args.FIX_POINT is not None:
        load_keys, episode_no, loss, acc = agent.load_pretrained_model(load_from=save_to_path, log_file=log_file)

    all_triplets_dict = {}
    poss_triplets_mask = None
    candidate_triplets_ids = []
    if agent.sample_number == 'None':  # do not sample:
        set_dir = config["graph_auto"]["data_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
            env.difficulty_level)
        with open(set_dir, 'r') as f:
            candidate_triplets = f.readlines()
            candidate_triplets = [triplet.replace('\n', '').split('$') for triplet in list(set(candidate_triplets))]
            poss_triplets_mask = agent.get_graph_adjacency_matrix([candidate_triplets])
        candidate_triplets_ids = [[_word_to_id(triplet[2], agent.relation2id),
                                   _word_to_id(triplet[0], agent.node2id),
                                   _word_to_id(triplet[1], agent.node2id)] for triplet in candidate_triplets]
    else:
        set_dir = config["graph_auto"]["data_path"] \
                  + "/difficulty_level_{0}/all_poss_set.txt".format(env.difficulty_level)
        all_triplets_dict = load_all_possible_set(set_dir=set_dir, log_file=log_file, )

    best_eval_diff_f1, best_eval_gen_f1 = 0.0, 0.0

    try:
        while (True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            target_graph_triplets, previous_graph_triplets, actions, observations = env.get_batch()
            input_adjacency_matrix = agent.get_graph_adjacency_matrix(previous_graph_triplets)
            output_adjacency_matrix = agent.get_graph_adjacency_matrix(target_graph_triplets)

            curr_load_data_batch_size = len(previous_graph_triplets)
            episode_no += curr_load_data_batch_size

            # the relations that have been changed by the action
            graph_diff_pos_sample_mask, graph_diff_neg_sample_mask, diff_triplet_index, \
            input_adjacency_matrix, output_adjacency_matrix, actions, pos_mask, neg_mask = \
                agent.get_diff_sample_masks(prev_adjs=input_adjacency_matrix,
                                            target_adjs=output_adjacency_matrix,
                                            actions=actions)

            # the general graph prediction
            if agent.sample_number == 'None':  # do not sample
                filter_mask_batch = np.repeat(poss_triplets_mask, curr_load_data_batch_size, axis=0)
                graph_negative_mask_agg = filter_mask_batch - output_adjacency_matrix
                tmp_min = np.min(graph_negative_mask_agg)
                assert tmp_min == 0
                tmp_max = np.max(graph_negative_mask_agg)
                assert tmp_max == 1
                sample_triplet_index_agg = [candidate_triplets_ids] * curr_load_data_batch_size
            else:
                graph_negative_mask_list, sample_number, sample_triplet_index_list = \
                    agent.get_graph_negative_sample_mask(adjs=output_adjacency_matrix,
                                                         triplets=target_graph_triplets,
                                                         sample_number=agent.sample_number,
                                                         all_triplets_dict=all_triplets_dict)
                sample_triplet_index_agg = merge_sample_triplet_index(sample_triplet_index_list)
                graph_negative_mask_agg = np.zeros(shape=output_adjacency_matrix.shape)
                for sample_index in range(sample_number):
                    graph_negative_mask = graph_negative_mask_list[sample_index]
                    graph_negative_mask_agg += graph_negative_mask

            if len(diff_triplet_index) == 0:
                continue
            pos_unique_count = {0.0: 0, 1.0: 0}
            neg_unique_count = {0.0: 0, 1.0: 0}

            c_skip_num = 0
            check_batch_size = len(graph_diff_pos_sample_mask)
            for b_idx in range(check_batch_size):
                pos_unique, pos_count = np.unique(graph_diff_pos_sample_mask[b_idx - c_skip_num], return_counts=True)
                # pos_sub_unique_count = {0.0: 0, 1.0: 0}
                pos_unique_count.update(dict(zip(pos_unique, pos_count)))
                # pos_sub_unique_count.update(dict(zip(pos_unique, pos_count)))
                neg_unique, neg_count = np.unique(graph_diff_neg_sample_mask[b_idx - c_skip_num], return_counts=True)
                # neg_sub_unique_count = {0.0: 0, 1.0: 0}
                neg_unique_count.update(dict(zip(neg_unique, neg_count)))

                # neg_sub_unique_count.update(dict(zip(neg_unique, neg_count)))
            if 'random' in config['general']['task']:
                input_adjacency_matrix = np.random.uniform(low=0.0, high=1.0, size=input_adjacency_matrix.shape)
            elif 'gtf' in config['general']['task']:
                input_adjacency_matrix = output_adjacency_matrix

            pre_loss, diff_loss, latent_loss, _, _, _, _ = agent.get_predict_dynamics_logits(
                input_adj_m=input_adjacency_matrix,
                actions=actions,
                observations=observations,
                output_adj_m=output_adjacency_matrix,
                hx=hx, cx=cx,
                graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                graph_negative_mask=graph_negative_mask_agg,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
            )

            # Update Model
            if debug_mode:
                weight = 1
            else:
                weight = weight_init + float(weight_upper_bound - weight_init) * batch_no / (agent.warmup_until) \
                    if batch_no <= agent.warmup_until else weight_upper_bound
            loss = pre_loss + diff_loss + weight * latent_loss
            agent.model.zero_grad()
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.clip_grad_norm)
            agent.optimizer.step()
            loss = to_np(loss)
            ave_train_loss.push(loss)

            if debug_mode:
                parameters_info = []
                for k, v in agent.model.named_parameters():
                    if v.grad is not None:
                        parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                    else:
                        parameters_info.append("{0}:{1}".format(k, v.grad))
                print(parameters_info, file=log_file, flush=True)

            if not debug_mode:
                # lr schedule
                if batch_no < agent.warmup_until:
                    cr = agent.init_learning_rate / math.log2(agent.warmup_until)
                    learning_rate = cr * math.log2(batch_no + 1)
                else:
                    learning_rate = agent.init_learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = learning_rate

            batch_no += 1
            if agent.report_frequency == 0 or (
                    episode_no % agent.report_frequency > (
                    episode_no - curr_load_data_batch_size) % agent.report_frequency):
                continue

            diff_eval_acc, gen_eval_acc, eval_loss, diff_train_loss, diff_train_acc, diff_train_dist, \
            gen_train_acc, gen_train_dist, gen_train_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            diff_tp_num_sum, diff_fp_num_sum, diff_tn_num_sum, diff_fn_num_sum = 0.0, 0.0, 0.0, 0.0
            gen_tp_num_sum, gen_fp_num_sum, gen_tn_num_sum, gen_fn_num_sum = 0.0, 0.0, 0.0, 0.0
            log_msg_append = ""

            if agent.run_eval:
                diff_list_train_acc, diff_list_train_loss, diff_list_score, \
                diff_list_soft_dist, diff_tp_num, diff_fp_num, diff_tn_num, diff_fn_num, hx, cx = \
                    compute_dynamic_predict_acc(agent=agent,
                                                graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                                                graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                                                graph_negative_mask=None,
                                                input_adj_m=input_adjacency_matrix,
                                                output_adj_m=output_adjacency_matrix,
                                                actions=actions,
                                                observations=observations,
                                                sample_triplet_index=diff_triplet_index,
                                                hx=hx,
                                                cx=cx,
                                                batch_mask=None,
                                                pos_mask=pos_mask,
                                                neg_mask=neg_mask,
                                                )
                diff_train_loss = np.mean(diff_list_train_loss)
                diff_train_acc = np.mean(diff_list_train_acc)
                diff_train_dist = np.mean(diff_list_soft_dist)
                diff_train_score = np.mean(diff_list_score)
                diff_tp_num_sum += diff_tp_num
                diff_fp_num_sum += diff_fp_num
                diff_tn_num_sum += diff_tn_num
                diff_fn_num_sum += diff_fn_num

                gen_list_train_acc, gen_list_train_loss, gen_list_score, \
                gen_list_soft_dist, gen_tp_num, gen_fp_num, gen_tn_num, gen_fn_num, hx, cx = \
                    compute_dynamic_predict_acc(agent=agent,
                                                graph_diff_pos_sample_mask=None,
                                                graph_diff_neg_sample_mask=None,
                                                graph_negative_mask=graph_negative_mask_agg,
                                                input_adj_m=input_adjacency_matrix,
                                                output_adj_m=output_adjacency_matrix,
                                                actions=actions,
                                                observations=observations,
                                                sample_triplet_index=sample_triplet_index_agg,
                                                hx=hx,
                                                cx=cx,
                                                batch_mask=None,
                                                pos_mask=pos_mask,
                                                neg_mask=neg_mask,
                                                )

                gen_train_loss = np.mean(gen_list_train_loss)
                gen_train_acc = np.mean(gen_list_train_acc)
                gen_train_dist = np.mean(gen_list_soft_dist)
                gen_train_score = np.mean(diff_list_score)
                gen_tp_num_sum += gen_tp_num
                gen_fp_num_sum += gen_fp_num
                gen_tn_num_sum += gen_tn_num
                gen_fn_num_sum += gen_fn_num

                if diff_tp_num_sum + diff_fp_num_sum > 0:
                    diff_train_precision = float(diff_tp_num_sum) / (diff_tp_num_sum + diff_fp_num_sum)
                else:
                    diff_train_precision = 0

                if diff_tp_num_sum + diff_fn_num_sum > 0:
                    diff_train_recall = float(diff_tp_num_sum) / (diff_tp_num_sum + diff_fn_num_sum)
                else:
                    diff_train_recall = 0

                if diff_train_recall > 0 and diff_train_precision > 0:
                    diff_train_f1 = 2 * diff_train_precision * diff_train_recall / (
                            diff_train_precision + diff_train_recall)
                else:
                    diff_train_f1 = 0

                if gen_tp_num_sum + gen_fp_num_sum > 0:
                    gen_train_precision = float(gen_tp_num_sum) / (gen_tp_num_sum + gen_fp_num_sum)
                else:
                    gen_train_precision = 0

                if gen_tp_num_sum + gen_fn_num_sum > 0:
                    gen_train_recall = float(gen_tp_num_sum) / (gen_tp_num_sum + gen_fn_num_sum)
                else:
                    gen_train_recall = 0

                if gen_train_recall > 0 and gen_train_precision > 0:
                    gen_train_f1 = 2 * gen_train_precision * gen_train_recall / (gen_train_precision + gen_train_recall)
                else:
                    gen_train_f1 = 0

                eval_loss, diff_eval_acc, gen_eval_acc, diff_eval_precision, gen_eval_precision, diff_eval_recall, \
                gen_eval_recall, diff_eval_f1, gen_eval_f1, \
                eval_soft_dist, log_msg_append = evaluate_graph_prediction(env=env,
                                                                           agent=agent,
                                                                           log_file=log_file,
                                                                           valid_test="valid",
                                                                           eval_max_counter=eval_max_counter,
                                                                           all_triplets_dict=all_triplets_dict,
                                                                           poss_triplets_mask=poss_triplets_mask,
                                                                           candidate_triplets_ids=candidate_triplets_ids
                                                                           )
                env.split_reset("train")

                if diff_eval_f1 + gen_eval_f1 > best_eval_gen_f1 + best_eval_diff_f1:
                    best_eval_diff_f1 = diff_eval_f1
                    best_eval_gen_f1 = gen_eval_f1
                    agent.save_model_to_path(save_to_path=save_to_path, episode_no=episode_no,
                                             eval_acc=(diff_eval_f1 + gen_eval_f1) / 2,
                                             eval_loss=eval_loss, log_file=log_file)
                    print("Saving best model so far! with Eval diff f1: {:2.3f} and gen f1: {:2.3f}".
                          format(best_eval_diff_f1, best_eval_gen_f1),
                          file=log_file, flush=True)
            time_2 = datetime.datetime.now()
            print(
                "Episode: {:3d} | time spent: {:s} | Diff Loss: {:2.3f} | Gen Loss: {:2.3f} | latent_loss: {:2.3f} | \n"
                "Train Diff Acc: {:2.3f} | Train Diff Precision: {:2.3f} | Train Diff recall: {:2.3f} |Train Diff f1: {:2.3f} | "
                "Train Diff Dist: {:2.3f} |Train Diff Loss: {:.3f} | Train Diff Score: {:2.8f} |\n"
                "Train Gen Acc: {:2.3f} | Train Gen Precision: {:2.3f} | Train Gen recall: {:2.3f} |Train Gen f1: {:2.3f} | "
                "Train Gen Dist: {:2.3f} |Train Gen Loss: {:.3f} |  Train Gen Score: {:2.8} | \n"
                "Eval Diff Acc: {:2.3f} | Eval Diff Precision: {:2.3f} | Eval Diff recall: {:2.3f} |Eval Diff f1: {:2.3f} | "
                "Eval Gen Acc: {:2.3f} |  Eval Gen Precision: {:2.3f} | Eval Gen recall: {:2.3f} |Eval Gen f1: {:2.3f} | \n "
                "Eval Loss: {:2.3f} | Eval_soft_dist: {:2.3f}| Learning rate: {:2.6f} | weight: {:2.3f} ".
                format(
                    episode_no, str(time_2 - time_1).rsplit(".")[0], diff_loss, pre_loss, latent_loss,
                    diff_train_acc, diff_train_precision, diff_train_recall, diff_train_f1, diff_train_dist,
                    diff_train_loss, diff_train_score,
                    gen_train_acc, gen_train_precision, gen_train_recall, gen_train_f1, gen_train_dist,
                    gen_train_loss, gen_train_score,
                    # str(pos_unique_count[float(1)]), str(neg_unique_count[float(1)]),
                    diff_eval_acc, diff_eval_precision, diff_eval_recall, diff_eval_f1,
                    gen_eval_acc, gen_eval_precision, gen_eval_recall, gen_eval_f1,
                    eval_loss, eval_soft_dist, learning_rate, weight) + log_msg_append,
                file=log_file, flush=True)
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
    # else:
    #     test(args)
