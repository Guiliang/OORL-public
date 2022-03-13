import datetime
import os

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)

import math
import torch
import numpy as np
from generic.state_prediction_dataset import SPData
from agent.agent import OORLAgent
from generic.data_utils import load_config, read_args, merge_sample_triplet_index
from generic.model_utils import HistoryScoreCache, to_np
from evaluate.evaluate_prediction import evaluate_graph_prediction, compute_auto_encoder_acc


def test(args):
    config, debug_mode, log_file_path, _ = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    eval_max_counter = None
    debug_msg = ''
    if debug_mode:
        debug_msg = '_debug'
        # config['general']['use_cuda'] = False
        config['general']['training']['batch_size'] = 3
        config['general']['checkpoint']['report_frequency'] = 1
        eval_max_counter = 2
        # config['general']['training']['sample_number'] = 1

    env = SPData(config)
    config['general']['checkpoint']['load_pretrained'] = False
    agent = OORLAgent(config, log_file, '')

    model_name = 'saved_model_ComplEx_seenGraph_Sep-20-2021_random_probing.pt'
    load_eval_model_path = agent.output_dir + agent.experiment_tag + "/{0}".format(model_name)
    loaded_keys = agent.load_pretrained_model(load_eval_model_path,
                                              log_file=log_file,
                                              load_partial_graph=False)

    eval_loss, _, gen_eval_acc, _, gen_eval_precision, _, gen_eval_recall, _, gen_eval_f1, eval_soft_dist, log_msg_append \
        = evaluate_graph_prediction(env, agent, log_file, "valid", eval_max_counter)

    print(" Eval Gen Acc: {:2.3f} |Eval Gen Precision: {:2.3f} |Eval Gen Recall: {:2.3f} |Eval Gen F1: {:2.3f} |"
          " eval_soft_dist: {:2.3f}  | Eval Loss: {:2.3f} |". \
          format(gen_eval_acc, gen_eval_precision, gen_eval_recall, gen_eval_f1, eval_soft_dist, eval_loss))


def train(args):
    time_1 = datetime.datetime.now()
    today = datetime.date.today()
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
    eval_max_counter = None
    if debug_mode:
        config['general']['training']['batch_size'] = 2
        config['general']['checkpoint']['report_frequency'] = 1
        config['general']['training']['optimizer']['learning_rate'] = 0.0005
        debug_msg += '_debug'
        # config['general']['training']['sample_number'] = 1
        eval_max_counter = 10
    env = SPData(config, seed=args.SEED, log_file=log_file)
    env.split_reset("train")
    agent = OORLAgent(config, log_file, debug_msg, seed=args.SEED)
    ave_train_loss = HistoryScoreCache(capacity=500)

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        loss_win = None
        eval_acc_win = None
        viz_loss, viz_eval_loss, viz_eval_acc = [], [], []

    episode_no = 0
    batch_no = 0
    save_to_path = agent.output_dir + agent.experiment_tag + "/saved_model_{0}_{1}Graph_{2}{3}.pt".format(
        agent.model.graph_decoding_method,
        config["graph_auto"]["graph_type"],
        today.strftime("%b-%d-%Y"),
        debug_msg)

    print("The model saving path is {0}".format(save_to_path),
          file=log_file, flush=True)

    best_eval_acc, best_training_loss_so_far = 0.0, 10000.0

    # target_graph_triplets, _, _, _ = env.get_batch()
    # print('-' * 100, file=log_file, flush=True)
    # print("*** Warning: Launching the overfitting experiment with few samples. ***", file=log_file, flush=True)
    # print('-' * 100, file=log_file, flush=True)
    try:
        while (True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            target_graph_triplets, _, _, _ = env.get_batch()
            curr_batch_size = len(target_graph_triplets)
            real_adjacency_matrix = agent.get_graph_adjacency_matrix(target_graph_triplets)
            if 'random' in config['general']['task']:
                input_adjacency_matrix = np.random.uniform(low=0.0, high=1.0, size=real_adjacency_matrix.shape)
            else:
                input_adjacency_matrix = real_adjacency_matrix
            graph_negative_mask_list, sample_number, sample_triplet_index_list = \
                agent.get_graph_negative_sample_mask(adjs=real_adjacency_matrix,
                                                     triplets=target_graph_triplets,
                                                     sample_number=agent.sample_number, )
            curr_load_data_batch_size = len(target_graph_triplets)
            sample_triplet_index_agg = merge_sample_triplet_index(sample_triplet_index_list)
            graph_negative_mask_agg = np.zeros(shape=real_adjacency_matrix.shape)
            for sample_index in range(sample_number):
                graph_negative_mask = graph_negative_mask_list[sample_index]
                graph_negative_mask_agg += graph_negative_mask

            unique, counts = np.unique(graph_negative_mask_agg, return_counts=True)
            dict(zip(unique, counts))

            loss, _, _ = agent.get_graph_autoencoder_logits(
                input_adj_m=input_adjacency_matrix,
                real_adj_m=real_adjacency_matrix,
                graph_negative_mask=graph_negative_mask_agg)

            # Update Model
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

            # lr schedule
            if batch_no < agent.warmup_until:
                cr = agent.init_learning_rate / math.log2(agent.warmup_until)
                learning_rate = cr * math.log2(batch_no + 1)
            else:
                learning_rate = agent.init_learning_rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = learning_rate

            episode_no += curr_load_data_batch_size
            batch_no += 1
            if agent.report_frequency == 0 or (
                    episode_no % agent.report_frequency > (
                    episode_no - curr_load_data_batch_size) % agent.report_frequency):
                continue

            if config["general"]["visdom"]:
                viz_loss.append(ave_train_loss.get_avg())

            eval_acc, eval_loss, eval_soft_dist, train_loss, train_acc, train_dist = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            log_msg_append = ""
            # sample_triplet_index = sample_triplet_index_list[0]
            # graph_negative_mask = graph_negative_mask_list[0]
            list_eval_acc, list_eval_loss, list_score, list_soft_dist, tp_num, tn_num, fp_num, fn_num = \
                compute_auto_encoder_acc(agent=agent,
                                         graph_negative_mask=graph_negative_mask_agg,
                                         input_adj_m=input_adjacency_matrix,
                                         real_adj_m=real_adjacency_matrix,
                                         sample_triplet_index=sample_triplet_index_agg)
            train_loss = np.mean(list_eval_loss)
            train_acc = np.mean(list_eval_acc)
            train_dist = np.mean(list_soft_dist)

            eval_loss, _, eval_acc, _, gen_eval_precision, _, gen_eval_recall, _, gen_eval_f1, \
            eval_soft_dist, log_msg_append = evaluate_graph_prediction(env, agent, log_file, "valid",
                                                                       eval_max_counter=eval_max_counter)

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                agent.save_model_to_path(save_to_path=save_to_path, episode_no=episode_no, eval_acc=eval_acc,
                                         eval_loss=eval_loss, log_file=log_file)
                print("Saving best model so far! with Eval acc : {:2.3f}".format(best_eval_acc),
                      file=log_file, flush=True)
            env.split_reset("train")
            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | n+p loss: {:2.3f} | \n"
                  "Train Acc: {:2.3f} | Train Dist: {:2.3f} |Train Loss: {:2.3f} | \n"
                  "Eval Acc: {:2.3f} | Eval Precision: {:2.3f} | Eval Recall: {:2.3f} | Eval f1: {:2.3f} |"
                  "Eval Dist: {:2.3f} | Eval Loss: {:2.3f}".format(
                episode_no, str(time_2 - time_1).rsplit(".")[0], loss,
                train_acc, train_dist, train_loss,
                eval_acc, gen_eval_precision, gen_eval_recall, gen_eval_f1,
                eval_soft_dist, eval_loss) + log_msg_append,
                  file=log_file, flush=True)

            # # plot using visdom
            # if config["general"]["visdom"]:
            #     viz_eval_acc.append(eval_acc)
            #     viz_eval_loss.append(eval_loss)
            #     viz_x = np.arange(len(viz_loss)).tolist()
            #     viz_eval_x = np.arange(len(viz_eval_acc)).tolist()
            #
            #     if loss_win is None:
            #         loss_win = viz.line(X=viz_x, Y=viz_loss,
            #                             opts=dict(title=agent.experiment_tag + "_loss"),
            #                             name="training loss")
            #         viz.line(X=viz_eval_x, Y=viz_eval_loss,
            #                  opts=dict(title=agent.experiment_tag + "_eval_loss"),
            #                  win=loss_win, update='append', name="eval loss")
            #     else:
            #         viz.line(X=[len(viz_loss) - 1], Y=[viz_loss[-1]],
            #                  opts=dict(title=agent.experiment_tag + "_loss"),
            #                  win=loss_win,
            #                  update='append', name="training loss")
            #         viz.line(X=[len(viz_eval_loss) - 1], Y=[viz_eval_loss[-1]],
            #                  opts=dict(title=agent.experiment_tag + "_eval_loss"),
            #                  win=loss_win, update='append', name="eval loss")
            #
            #     if eval_acc_win is None:
            #         eval_acc_win = viz.line(X=viz_eval_x, Y=viz_eval_acc,
            #                                 opts=dict(title=agent.experiment_tag + "_eval_acc"),
            #                                 name="eval accuracy")
            #     else:
            #         viz.line(X=[len(viz_eval_acc) - 1], Y=[viz_eval_acc[-1]],
            #                  opts=dict(title=agent.experiment_tag + "_eval_acc"),
            #                  win=eval_acc_win,
            #                  update='append', name="eval accuracy")
            #
            # # write accuracies down into file
            # _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
            #                  "loss": str(ave_train_loss.get_avg()),
            #                  "eval loss": str(eval_loss),
            #                  "eval accuracy": str(eval_acc)})
            # with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            #     outfile.write(_s + '\n')
            #     outfile.flush()

    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        log_file.close()
        print('--------------------------------------------')
        print('Exiting from training early...')
    # if agent.run_eval:
    #     if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
    #         print('Evaluating on test set and saving log...')
    #         agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
    #     _, _ = evaluate.evaluate_state_prediction(env, agent, "test", verbose=True)


if __name__ == '__main__':
    args = read_args()
    if int(args.TRAIN_FLAG):
        train(args)
    else:
        test(args)
