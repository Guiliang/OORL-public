import codecs
import copy
import itertools
import spacy
import random
import torch
import numpy as np
import traceback
from textworld import EnvInfos
import torch.nn.functional as F

from generic import memory_buffer
from generic.data_utils import load_graph_ids, _word_to_id, _words_to_ids, max_len, pad_sequences, preproc, \
    compute_mask, NegativeLogLoss
from generic.model_utils import to_pt, kl_divergence, LinearSchedule, to_np, ez_gather_dim_1, masked_mean
from model.graph_model import KG_Manipulation


class OORLAgent:
    def __init__(self, config, log_file, debug_msg,
                 seed=None,
                 skip_load=False,
                 split_optimizer=False,
                 init_optimizer=True):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # read the settings
        self.assign_agent_config(config=config, log_file=log_file)

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.model = KG_Manipulation(config=config, word_vocab=self.word_vocab,
                                     node_vocab=self.node_vocab, relation_vocab=self.relation_vocab,
                                     task_name=self.task, log_file=log_file)
        # self.online_net.train()
        if self.use_cuda:
            self.model.cuda()

        # load model from checkpoint
        if self.load_pretrained and not skip_load:
            self.loading_model_components(log_file)

        assert self.step_rule == 'radam'
        from generic.radam import RAdam

        if not skip_load and init_optimizer:  # init optimizer, which is not necessary during testing
            if 'rl_planning' in self.task:
                self.optimizers = {'DQN': None, 'reward_model': None, 'transition_model': None}
                param_active_key_words = {'DQN': ['q_online_net', 'encoder_for_q_values'],
                                          'reward_model': ['reward'],
                                          'transition_model': ['dynamic_model', 'decode_graph', 'action_encoder',
                                                               'obs_encoder']}
                all_active_key_words = set()
                for key_words in param_active_key_words.values():
                    all_active_key_words = all_active_key_words.union(set(key_words))
            elif 'unsupervised' in self.task:
                self.optimizers = {'reward_model': None, 'obs_gen_model': None, 'transition_model': None}
                param_active_key_words = {'reward_model': ['reward'],
                                          'obs_gen_model': ['obs_gen'],
                                          'goal_gen_model': ['goal_gen'],
                                          'transition_model': ['dynamic_model', 'decode_graph', 'action_encoder',
                                                               'obs_encoder', 'goal_encoder']}
                all_active_key_words = set()
                for key_words in param_active_key_words.values():
                    all_active_key_words = all_active_key_words.union(set(key_words))
                param_frozen_list, param_active_list = \
                    self.handle_model_parameters(
                        fix_keywords=self.fix_parameters_keywords.union(all_active_key_words) - set(
                            param_active_key_words['goal_gen_model']),
                        model_name='goal_gen_model',
                        log_file=log_file,
                        set_require_grad=False,
                    )
                self.goal_gen_optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                                 {'params': param_active_list,
                                                  'lr': config['general']['training']['optimizer']['learning_rate']}],
                                                lr=config['general']['training']['optimizer']['learning_rate'])
            else:
                self.optimizers = None
                param_active_key_words = {}
                all_active_key_words = set()

            if 'rl_planning' in self.task:
                for model_name in self.optimizers.keys():
                    param_frozen_list, param_active_list = \
                        self.handle_model_parameters(
                            fix_keywords=self.fix_parameters_keywords.union(all_active_key_words) - set(
                                param_active_key_words[model_name]),
                            model_name=model_name,
                            log_file=log_file,
                            set_require_grad=False,
                        )
                    optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                       {'params': param_active_list,
                                        'lr': config['general']['training']['optimizer']['learning_rate']}],
                                      lr=config['general']['training']['optimizer']['learning_rate'])
                    self.optimizers.update({model_name: optimizer})
            else:
                param_frozen_list, param_active_list = \
                    self.handle_model_parameters(fix_keywords=self.fix_parameters_keywords,
                                                 model_name=self.task,
                                                 log_file=log_file,
                                                 set_require_grad=True)
                self.optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                        {'params': param_active_list,
                                         'lr': config['general']['training']['optimizer']['learning_rate']}],
                                       lr=config['general']['training']['optimizer']['learning_rate'])

    def assign_agent_config(self, config, log_file):
        self.task = config['general']['task']
        print('Task name: {0}'.format(self.task), file=log_file, flush=True)
        self.step_rule = config['general']['training']['optimizer']['step_rule']
        self.init_learning_rate = config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = config['general']['training']['optimizer']['clip_grad_norm']
        self.warmup_until = config['general']['training']['optimizer']['learning_rate_warmup_until']
        self.fix_parameters_keywords = list(set(config['general']['training']['fix_parameters_keywords']))
        self.batch_size = config['general']['training']['batch_size']
        self.max_episode = config['general']['training']['max_episode']
        self.smoothing_eps = config['general']['training']['smoothing_eps']
        self.patience = config['general']['training']['patience']
        self.sample_number = config['general']['training']['sample_number']

        self.output_dir = config['general']['checkpoint']['output_dir']
        self.experiment_tag = config['general']['checkpoint']['experiment_tag']
        self.save_frequency = config['general']['checkpoint']['save_frequency']
        self.report_frequency = config['general']['checkpoint']['report_frequency']
        self.load_pretrained = config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = config['general']['checkpoint']['load_from_tag']
        self.load_from_label = config['general']['checkpoint']['load_from_label']
        self.data_label = config['general']['checkpoint']['data_label']
        self.fix_loaded_parameters = config['general']['checkpoint']['fix_loaded_parameters']
        self.load_graph_generation_model_from_tag = config['general']['checkpoint'][
            'load_graph_generation_model_from_tag']
        self.load_partial_parameter_keywords = list(
            set(config['general']['checkpoint']['load_partial_parameter_keywords']))

        self.run_eval = config['general']['evaluate']['run_eval']
        self.eval_g_belief = config['general']['evaluate']['g_belief']
        self.eval_batch_size = config['general']['evaluate']['batch_size']
        self.max_target_length = config['general']['evaluate']['max_target_length']

        # Set the random seed manually for reproducibility.
        self.random_seed = config['general']['random_seed']

        if "dynamic_predict_ims" in self.task or "graph_autoenc" in self.task:
            self.graph_type = config['graph_auto']['graph_type']
            self.difficulty_level = config['graph_auto']['difficulty_level']
        elif 'reward_prediction' in self.task or "unsupervised" in self.task:
            self.graph_type = config['reward_prediction']['graph_type']
            self.difficulty_level = config['reward_prediction']['difficulty_level']
            self.apply_real_goal = config['reward_prediction']['apply_real_goal']
            self.reward_loss_type = config['reward_prediction']['loss_type']
            self.if_reward_dropout = config['reward_prediction']['if_reward_dropout']
            print("Reward loss is: {0}".format(self.reward_loss_type), file=log_file, flush=True)
            if self.apply_real_goal == 'None':
                print("Goal Type: No Goal",
                      file=log_file, flush=True)
            else:
                print("Goal Type: {0}".format('Ground Truth' if self.apply_real_goal else "Predicted"),
                      file=log_file, flush=True)
        if 'unsupervised' in self.task:
            self.if_condition_decoder = config['reward_prediction']['if_condition_decoder']
            print("Apply{0} condition to decoder".format('' if self.if_condition_decoder else ' no'),
                  file=log_file, flush=True)

        if 'rl_planning' in self.task:
            self.graph_type = config['rl']['graph_type']
            self.difficulty_level = config['rl']['difficulty_level']
            self.max_scores = config['rl']['max_scores']
            self.use_negative_reward = config['rl']['training']['use_negative_reward']
            self.max_nb_steps_per_episode = config['rl']['training']['max_nb_steps_per_episode']
            self.learn_start_from_this_episode = config['rl']['training']['learn_start_from_this_episode']
            self.target_net_update_frequency = config['rl']['training']['target_net_update_frequency']
            self.apply_goal_constraint = config['rl']['training']['apply_goal_constraint']
            self.eval_max_nb_steps_per_episode = config['rl']['evaluate']['max_nb_steps_per_episode']

            self.c_puct = config['rl']['planing']['c_puct']
            self.simulations_num = config['rl']['planing']['simulations_num']
            self.planner_name = config['rl']['planing']['planner']
            self.discount_rate = config['rl']['planing']['discount_rate']
            self.random_move_prob_add = config['rl']['planing']['random_move_prob_add']
            self.max_search_depth = float(config['rl']['planing']['max_search_depth'])
            self.load_extractor = float(config['rl']['planing']['load_extractor'])

            self.replay_sample_history_length = config['rl']['replay']['replay_sample_history_length']
            self.replay_sample_update_from = config['rl']['replay']['replay_sample_update_from']
            # replay buffer and updates
            self.buffer_reward_threshold = config['rl']['replay']['buffer_reward_threshold']
            self.prioritized_replay_beta = config['rl']['replay']['prioritized_replay_beta']
            self.accumulate_reward_from_final = config['rl']['replay']['accumulate_reward_from_final']
            self.prioritized_replay_eps = config['rl']['replay']['prioritized_replay_eps']
            self.count_reward_lambda = config['rl']['replay']['count_reward_lambda']
            self.discount_gamma_count_reward = config['rl']['replay']['discount_gamma_count_reward']
            self.graph_reward_lambda = config['rl']['replay']['graph_reward_lambda']
            self.graph_reward_type = config['rl']['replay']['graph_reward_type']
            self.discount_gamma_graph_reward = config['rl']['replay']['discount_gamma_graph_reward']
            self.discount_gamma_game_reward = config['rl']['replay']['discount_gamma_game_reward']
            self.replay_batch_size = config['rl']['replay']['replay_batch_size']
            self.update_per_k_game_steps = config['rl']['replay']['update_per_k_game_steps']
            self.multi_step = config['rl']['replay']['multi_step']

            self.dqn_memory = memory_buffer.PrioritizedReplayMemory(
                config['rl']['replay']['replay_memory_capacity'],
                priority_fraction=config['rl']['replay']['replay_memory_priority_fraction'],
                discount_gamma_game_reward=self.discount_gamma_game_reward,
                discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                discount_gamma_count_reward=self.discount_gamma_count_reward,
                accumulate_reward_from_final=self.accumulate_reward_from_final,
                supervised_rl_flag=False if 'unsupervised' in self.task else True)

            self.model_memory = memory_buffer.BalancedTransitionMemory(
                capacity=int(config['rl']['replay']['replay_memory_capacity'] / 2),
                seed=self.random_seed,
                supervised_rl_flag=False if 'unsupervised' in self.task else True
            )

            self.epsilon_anneal_episodes = config['rl']['epsilon_greedy']['epsilon_anneal_episodes']
            self.epsilon_anneal_from = config['rl']['epsilon_greedy']['epsilon_anneal_from']
            self.epsilon_anneal_to = config['rl']['epsilon_greedy']['epsilon_anneal_to']
            self.min_unexplore_rate = config['rl']['epsilon_greedy']['min_unexplore_rate']
            self.epsilon = self.epsilon_anneal_from
            self.epsilon_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes,
                                                    initial_p=self.epsilon_anneal_from, final_p=self.epsilon_anneal_to)
            self.unexplore_rate = self.epsilon_anneal_from
            self.unexplore_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes,
                                                      initial_p=self.epsilon_anneal_from,
                                                      final_p=self.min_unexplore_rate)
            self.max_nb_steps_per_episode_scale = self.max_nb_steps_per_episode
            self.max_nb_steps_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes,
                                                         initial_p=self.max_nb_steps_per_episode,
                                                         final_p=self.max_nb_steps_per_episode)
            self.beta_scheduler = LinearSchedule(schedule_timesteps=self.max_episode,
                                                 initial_p=self.prioritized_replay_beta, final_p=1.0)

        if torch.cuda.is_available():
            if not config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        # word vocab
        self.word_vocab = []
        with codecs.open("../source/vocabularies/word_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i

        self.node_vocab, self.node2id, self.relation_vocab, \
        self.relation2id, self.origin_relation_number = load_graph_ids()

    def loading_model_components(self, log_file):
        """
        load different pre-trained models
        """
        if len(self.load_from_label[0]) > 0:  # load the graph encoder and decoder
            load_pretrain_auto_encoder_path = self.output_dir + self.load_from_tag[
                0] + "/saved_model_{0}_{1}Graph_{2}.pt".format(
                self.model.graph_decoding_method,
                self.graph_type,
                self.load_from_label[0])
            loaded_keys, _, _, _ = self.load_pretrained_model(load_pretrain_auto_encoder_path,
                                                              log_file=log_file,
                                                              load_partial_graph=False)
            if self.fix_loaded_parameters:  # fix the load parameters
                self.fix_parameters_keywords += loaded_keys
            print("Loaded keys are : {0}\n".format(loaded_keys), file=log_file, flush=True)

        if len(self.load_from_label[1]) > 0:
            load_pretrain_dynamic_model_path = self.output_dir + self.load_from_tag[
                1] + "/saved_model_dynamic_{3}_{0}_{4}_{6}_{2}.pt".format(
                self.model.graph_decoding_method,
                self.graph_type,
                self.load_from_label[1],
                self.model.dynamic_model_type,
                self.model.dynamic_model_mechanism,
                self.difficulty_level,
                self.model.dynamic_loss_type,
            )
            loaded_keys, _, _, _ = self.load_pretrained_model(load_pretrain_dynamic_model_path,
                                                              log_file=log_file,
                                                              load_partial_graph=False)
            if self.fix_loaded_parameters:  # fix the load parameters
                self.fix_parameters_keywords += loaded_keys
            print("Loaded keys are : {0}\n".format(loaded_keys), file=log_file, flush=True)

        if len(self.load_from_label[2]) > 0:

            if 'unsupervised' in self.task:
                load_pretrain_dynamic_model_path = self.output_dir + \
                                                   self.load_from_tag[2] + "/difficulty_level_{4}/" \
                                                                           "saved_model_unsupervised_latent_dynamic" \
                                                                           "_{0}_{1}{2}_dec-ComplEx_{3}.pt".format(
                    self.model.dynamic_model_type,
                    self.model.dynamic_model_mechanism,
                    '_cond' if self.if_condition_decoder else '',
                    self.load_from_label[2],
                    self.difficulty_level)
                # saved_model_unsupervised_latent_dynamic_linear_all-independent_dec-ComplEx_df-3_weight-1_Jul-01-2021.pt

            elif self.model.reward_predictor_apply_rnn:
                load_pretrain_dynamic_model_path = self.output_dir + \
                                                   self.load_from_tag[2] + "/difficulty_level_{3}/" \
                                                                           "saved_model_Dynamic_{0}_{1}_" \
                                                                           "Reward_Predictor_Goal_RNN_{2}.pt".format(
                    self.model.dynamic_model_mechanism,
                    self.model.dynamic_loss_type,
                    self.load_from_label[2],
                    self.difficulty_level)
            else:
                load_pretrain_dynamic_model_path = self.output_dir + \
                                                   self.load_from_tag[2] + "/difficulty_level_{1}/" \
                                                                           "saved_model_Dynamic_Reward_Predictor" \
                                                                           "_Goal_Linear_{0}.pt".format(
                    self.load_from_label[2],
                    self.difficulty_level)

            loaded_keys, _, _, _ = self.load_pretrained_model(load_pretrain_dynamic_model_path,
                                                              log_file=log_file,
                                                              load_partial_graph=False)
            if self.fix_loaded_parameters:  # fix the load parameters
                self.fix_parameters_keywords += loaded_keys
            print("Loaded keys are : {0}\n".format(loaded_keys), file=log_file, flush=True)

        self.fix_parameters_keywords = set(self.fix_parameters_keywords)

    def handle_model_parameters(self, fix_keywords, model_name, log_file, set_require_grad):
        """determine which parameters should be fixed"""
        # exclude some parameters from optimizer
        param_frozen_list = []  # should be changed into torch.nn.ParameterList()
        param_active_list = []  # should be changed into torch.nn.ParameterList()
        fixed_parameters_keys = []
        active_parameters_keys = []
        parameters_info = []

        for k, v in self.model.named_parameters():
            keep_this = True
            size = torch.numel(v)
            parameters_info.append("{0}:{1}".format(k, size))
            for keyword in fix_keywords:
                if keyword in k:
                    param_frozen_list.append(v)
                    if set_require_grad:
                        v.requires_grad = False  # fix the parameters https://pytorch.org/docs/master/notes/autograd.html
                    keep_this = False
                    fixed_parameters_keys.append(k)
                    break
            if keep_this:
                param_active_list.append(v)
                active_parameters_keys.append(k)
        print('-' * 30 + '{0} Optimizer'.format(model_name) + '-' * 30, file=log_file, flush=True)
        print("Fixed parameters are: {0}".format(str(fixed_parameters_keys)), file=log_file, flush=True)
        print("Active parameters are: {0}".format(str(active_parameters_keys)), file=log_file, flush=True)
        # print(parameters_info, file=log_file, flush=True)
        param_frozen_list = torch.nn.ParameterList(param_frozen_list)
        param_active_list = torch.nn.ParameterList(param_active_list)
        print('-' * 60, file=log_file, flush=True)

        return param_frozen_list, param_active_list

    def load_pretrained_model(self, load_from, log_file,
                              load_partial_graph=True,
                              load_running_records=False):
        """
        Load pretrained checkpoint from file.

        Arguments:
            :param load_from: File name of the pretrained model checkpoint.
            :param log_file: the log
        """
        print("loading model from %s\n" % (load_from), file=log_file, flush=True)
        try:
            if self.use_cuda:
                checkpoint = torch.load(load_from)
            else:
                checkpoint = torch.load(load_from, map_location='cpu')

            model_checkpoints = checkpoint['model_state_dict']
            # optimizer_checkpoints = checkpoint['optimizer_state_dict']
            epoch = checkpoint['epoch']
            loss = checkpoint['eval_loss']
            acc = checkpoint['eval_acc']
            if 'running_game_points' in checkpoint.keys():
                running_game_points = checkpoint['running_game_points']
            else:
                running_game_points = None
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in model_checkpoints.items() if k in model_dict}
            omitted_dict = {k: v for k, v in model_checkpoints.items() if k not in model_dict}
            unload_dict = {k: v for k, v in model_dict.items() if k not in model_checkpoints}
            if load_partial_graph and len(self.load_partial_parameter_keywords) > 0:
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_partial_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            loss = float('inf') if loss is None else loss
            print("Successfully load model with epoch:{:2.3f}, eval loss:{:2.3f}, "
                  "eval acc:{:2.3f}, and parameters:".format(epoch, loss, acc), file=log_file, flush=True)
            load_keys = [key for key in pretrained_dict]
            omit_keys = [key for key in omitted_dict]
            print("Loaded model parameters are:" + ", ".join(load_keys), file=log_file, flush=True)
            print("Omitted model parameters are:" + ", ".join(omit_keys), file=log_file, flush=True)
            print("Omitted checkpoint parameters are:" + ", ".join(unload_dict), file=log_file, flush=True)
            print("--------------------------", file=log_file, flush=True)

            if load_running_records:
                return load_keys, epoch, loss, acc, running_game_points
            else:
                return load_keys, epoch, loss, acc
        except Exception:
            traceback.print_exc()
            print("Failed to load checkpoint...", file=log_file, flush=True)
            if load_running_records:
                return [], 0, 0, 0, {}
            else:
                return [], 0, 0, 0

    def get_graph_adjacency_matrix(self, triplets):
        """
        transfer triplets to the adj matrix
        """
        adj = np.zeros((len(triplets), len(self.relation_vocab), len(self.node_vocab), len(self.node_vocab)),
                       dtype="float32")
        for b in range(len(triplets)):
            node_exists = set()
            for t in triplets[b]:
                node1, node2, relation = t
                assert node1 in self.node_vocab, node1 + " is not in node vocab"
                assert node2 in self.node_vocab, node2 + " is not in node vocab"
                assert relation in self.relation_vocab, relation + " is not in relation vocab"
                node1_id = _word_to_id(node1, self.node2id)
                node2_id = _word_to_id(node2, self.node2id)
                relation_id = _word_to_id(relation, self.relation2id)
                adj[b][relation_id][node1_id][node2_id] = 1.0
                adj[b][relation_id + self.origin_relation_number][node2_id][node1_id] = 1.0
                node_exists.add(node1_id)
                node_exists.add(node2_id)
            # self relation
            for node_id in list(node_exists):
                adj[b, -1, node_id, node_id] = 1.0
        return adj

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.model.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.model.eval()

    def get_diff_sample_masks(self, prev_adjs,
                              target_adjs, actions,
                              observations=None, mask=None):
        """Actions have limited impact on graph, so we compute the masks of updated pos/neg triplets after
        performing the action. The lost should specially focus on these updated triplets."""
        batch_size = len(prev_adjs)
        graph_diff_pos_sample_mask = np.zeros((batch_size,
                                               len(self.relation_vocab),
                                               len(self.node_vocab),
                                               len(self.node_vocab)), dtype="float32")
        graph_diff_neg_sample_mask = np.zeros((batch_size,
                                               len(self.relation_vocab),
                                               len(self.node_vocab),
                                               len(self.node_vocab)), dtype="float32")
        triplet_index = []
        skip_num = 0
        pos_mask = []
        neg_mask = []
        for b_idx in range(batch_size):
            prev_adj = prev_adjs[b_idx - skip_num]
            target_adj = target_adjs[b_idx - skip_num]
            diff_adj = target_adj - prev_adj
            post_diff_index = np.where(diff_adj == 1)
            neg_diff_index = np.where(diff_adj == -1)
            if len(neg_diff_index[0]) == 0:
                neg_mask.append(0)
            else:
                neg_mask.append(1)
            if len(post_diff_index[0]) == 0:
                pos_mask.append(0)
            else:
                pos_mask.append(1)
            triplet_index.append([])
            for i in range(len(post_diff_index[0])):
                relation_id = post_diff_index[0][i]
                node1_id = post_diff_index[1][i]
                node2_id = post_diff_index[2][i]
                graph_diff_pos_sample_mask[b_idx - skip_num][relation_id][node1_id][node2_id] = 1
                triplet_index[b_idx - skip_num].append([relation_id, node1_id, node2_id])

            for i in range(len(neg_diff_index[0])):
                relation_id = neg_diff_index[0][i]
                node1_id = neg_diff_index[1][i]
                node2_id = neg_diff_index[2][i]
                graph_diff_neg_sample_mask[b_idx - skip_num][relation_id][node1_id][node2_id] = 1
                triplet_index[b_idx - skip_num].append([relation_id, node1_id, node2_id])

            assert np.sum(graph_diff_pos_sample_mask[b_idx - skip_num]) \
                   + np.sum(graph_diff_neg_sample_mask[b_idx - skip_num]) \
                   == len(triplet_index[b_idx - skip_num])

        pos_mask = to_pt(np.asarray(pos_mask), self.use_cuda)
        neg_mask = to_pt(np.asarray(neg_mask), self.use_cuda)
        if observations is not None and mask is not None:
            return graph_diff_pos_sample_mask, graph_diff_neg_sample_mask, triplet_index, \
                   prev_adjs, target_adjs, actions, observations, mask, pos_mask, neg_mask
        else:
            return graph_diff_pos_sample_mask, graph_diff_neg_sample_mask, triplet_index, \
                   prev_adjs, target_adjs, actions, pos_mask, neg_mask

    def get_graph_negative_sample_mask(self, adjs, triplets, sample_number, all_triplets_dict={}):
        """
        generate negative samples of triplets for training
        """
        graph_negative_sample_mask_list = [np.zeros((len(adjs),
                                                     len(self.relation_vocab),
                                                     len(self.node_vocab),
                                                     len(self.node_vocab)), dtype="float32") for i in
                                           range(sample_number)]
        sample_triplet_index_list = []  # [sample_number, batch_size, triplet_num, 3]
        for i in range(sample_number):
            sample_triplet_index_list.append([])

        for batch_idx in range(len(triplets)):
            adj = adjs[batch_idx]
            node_exists = set()

            for i in range(sample_number):
                sample_triplet_index_list[i].append([])

            for t in triplets[batch_idx]:
                node1, node2, relation = t
                assert node1 in self.node_vocab, node1 + " is not in node vocab"
                assert node2 in self.node_vocab, node2 + " is not in node vocab"
                assert relation in self.relation_vocab, relation + " is not in relation vocab"
                node1_id = _word_to_id(node1, self.node2id)
                node2_id = _word_to_id(node2, self.node2id)
                relation_id = _word_to_id(relation, self.relation2id)
                # for picking negative samples
                random_object_node_ids = random.sample(range(len(self.node_vocab)),
                                                       len(self.node_vocab))  # negative samples
                random_object_node_ids.remove(node2_id)
                random_object_relation_ids = random.sample(range(int(len(self.relation_vocab) / 2)),
                                                           int(len(self.relation_vocab) / 2))

                # negative samples
                all_node_relation_combination = []
                if len(all_triplets_dict) == 0:
                    for item in itertools.product(random_object_node_ids, random_object_relation_ids):
                        all_node_relation_combination.append([node1_id, item[0], item[1]])
                else:
                    candidate_triplets = list(all_triplets_dict[node1])
                    random_candidate_index = random.sample(range(len(candidate_triplets)), len(candidate_triplets))
                    for index in random_candidate_index:
                        items = candidate_triplets[index].split('$')
                        candidate_sub_node_id = _word_to_id(items[0], self.node2id)
                        candidate_obj_node_id = _word_to_id(items[1], self.node2id)
                        candidate_relation_id = _word_to_id(items[2], self.relation2id)
                        all_node_relation_combination.append(
                            [candidate_sub_node_id, candidate_obj_node_id, candidate_relation_id])

                    candidate_triplets = list(all_triplets_dict[node2])
                    random_candidate_index = random.sample(range(len(candidate_triplets)), len(candidate_triplets))
                    # print(candidate_triplets)
                    for index in random_candidate_index:
                        items = candidate_triplets[index].split('$')
                        candidate_sub_node_id = _word_to_id(items[0], self.node2id)
                        candidate_obj_node_id = _word_to_id(items[1], self.node2id)
                        candidate_relation_id = _word_to_id(items[2], self.relation2id)
                        all_node_relation_combination.append(
                            [candidate_sub_node_id, candidate_obj_node_id, candidate_relation_id])

                random_counter = 0
                for item in all_node_relation_combination:
                    [random_subject_node_id, random_object_node_id, random_relation_id] = item
                    if not adj[random_relation_id][random_subject_node_id][random_object_node_id] and \
                            not adj[random_relation_id + self.origin_relation_number][random_object_node_id][
                                random_subject_node_id]:
                        graph_negative_sample_mask_list[random_counter][
                            batch_idx, random_relation_id, random_subject_node_id, random_object_node_id] = 1.0
                        graph_negative_sample_mask_list[random_counter][
                            batch_idx, random_relation_id + self.origin_relation_number,
                            random_object_node_id, random_subject_node_id] = 1.0
                        sample_triplet_index_list[random_counter][batch_idx].append(
                            [relation_id, node1_id, node2_id])
                        sample_triplet_index_list[random_counter][batch_idx].append(
                            [random_relation_id, random_subject_node_id, random_object_node_id])
                        sample_triplet_index_list[random_counter][batch_idx].append(
                            [relation_id + self.origin_relation_number, node2_id, node1_id])
                        sample_triplet_index_list[random_counter][batch_idx].append(
                            [random_relation_id + self.origin_relation_number, random_object_node_id,
                             random_subject_node_id])
                        node_exists.add(node1_id)
                        node_exists.add(node2_id)
                        node_exists.add(random_subject_node_id)
                        node_exists.add(random_object_node_id)
                        random_counter += 1
                    else:
                        continue
                    if random_counter == sample_number:
                        break
        return graph_negative_sample_mask_list, sample_number, sample_triplet_index_list

    def get_predict_dynamics_logits(self, input_adj_m, actions, output_adj_m, hx, cx,
                                    graph_diff_pos_sample_mask,
                                    graph_diff_neg_sample_mask,
                                    graph_negative_mask,
                                    observations,
                                    pos_mask=None,
                                    neg_mask=None,
                                    batch_mask=None,
                                    weight=1,
                                    if_loss_mean=True):
        """
        return the loss for dynamic training
        """

        real_output_adj = to_pt(output_adj_m, self.use_cuda, type='float')

        if self.model.dynamic_loss_type == 'label':  # loss on hard label
            predicted_encodings, hx_new, cx_new, attn_mask, \
            input_node_name, input_relation_name, node_encodings, relation_encodings, \
            action_encodings_sequences, action_mask = \
                self.compute_updated_dynamics(input_adj_m, actions, observations, hx, cx)
            predict_output_adj = self.model.decode_graph(predicted_encodings, relation_encodings)
            pred_loss, diff_loss = self.dynamic_graph_label_loss(predict_output_adj=predict_output_adj,
                                                                 real_output_adj=real_output_adj,
                                                                 graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                                                                 pos_mask=pos_mask,
                                                                 graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                                                                 neg_mask=neg_mask,
                                                                 batch_mask=batch_mask,
                                                                 graph_negative_mask=graph_negative_mask,
                                                                 )
            if if_loss_mean:
                return torch.mean(pred_loss * weight), torch.mean(diff_loss), \
                       torch.mean(to_pt(np.zeros([len(predict_output_adj)]), self.use_cuda, type='float')), \
                       predict_output_adj, real_output_adj, hx_new, cx_new
            else:
                return pred_loss * weight, diff_loss, \
                       to_pt(np.zeros([len(predict_output_adj)]), self.use_cuda, type='float'), \
                       predict_output_adj, real_output_adj, hx_new, cx_new

        elif self.model.dynamic_loss_type == 'latent':  # loss on the node encodings
            latent_loss, predicted_encodings_post, predicted_encodings_prior, relation_encodings, node_mask, _, _ = \
                self.dynamic_latent_loss(actions=actions,
                                         observations=observations,
                                         input_adj_m=input_adj_m,
                                         hx=hx,
                                         cx=cx,
                                         if_clamp_latent_loss=True if 'single' in self.model.dynamic_model_mechanism else False
                                         )
            recon_output_adj = self.model.decode_graph(predicted_encodings_post, relation_encodings)
            pred_loss, diff_loss = self.dynamic_graph_label_loss(predict_output_adj=recon_output_adj,
                                                                 real_output_adj=real_output_adj,
                                                                 graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
                                                                 pos_mask=pos_mask,
                                                                 graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
                                                                 neg_mask=neg_mask,
                                                                 batch_mask=batch_mask,
                                                                 graph_negative_mask=graph_negative_mask,
                                                                 )
            if if_loss_mean:
                return torch.mean(pred_loss * weight), torch.mean(diff_loss), torch.mean(latent_loss), \
                       recon_output_adj, real_output_adj, None, None
            else:
                return pred_loss * weight, diff_loss, latent_loss, \
                       recon_output_adj, real_output_adj, None, None
        else:
            raise ValueError("Unknown dynamic_loss_type {0}".format(self.model.dynamic_loss_type))

    def dynamic_graph_label_loss(self, predict_output_adj, real_output_adj,
                                 graph_diff_pos_sample_mask, pos_mask,
                                 graph_diff_neg_sample_mask, neg_mask,
                                 batch_mask, graph_negative_mask):
        """
        get the resampled loss for graph prediction
        """
        predict_output_adj = torch.clamp(predict_output_adj, min=0, max=1)
        if not torch.max(predict_output_adj) <= 1:
            print(predict_output_adj)
        assert torch.max(predict_output_adj) <= 1
        assert torch.min(predict_output_adj) >= 0
        all_loss = self.model.bce_loss(input=predict_output_adj, target=real_output_adj)
        if graph_diff_pos_sample_mask is not None:
            graph_diff_pos_sample_mask = to_pt(graph_diff_pos_sample_mask, self.use_cuda,
                                               type='float')  # type='long'
            graph_diff_neg_sample_mask = to_pt(graph_diff_neg_sample_mask, self.use_cuda,
                                               type='float')  # type='long'
            diff_pos_loss = all_loss * graph_diff_pos_sample_mask
            diff_neg_loss = all_loss * graph_diff_neg_sample_mask

            ones = torch.ones([len(predict_output_adj)])
            if self.use_cuda:
                ones = ones.cuda()
            diff_pos_loss_sum = torch.sum(diff_pos_loss, dim=(1, 2, 3)) * pos_mask
            diff_pos_loss_num = torch.where(pos_mask > 0, torch.sum(graph_diff_pos_sample_mask, dim=(1, 2, 3)), ones)
            diff_pos_loss = diff_pos_loss_sum / diff_pos_loss_num
            diff_neg_loss_sum = torch.sum(diff_neg_loss, dim=(1, 2, 3)) * neg_mask
            diff_neg_loss_num = torch.where(neg_mask > 0, torch.sum(graph_diff_neg_sample_mask, dim=(1, 2, 3)), ones)
            diff_neg_loss = diff_neg_loss_sum / diff_neg_loss_num
            diff_loss = diff_neg_loss + diff_pos_loss
            if batch_mask is not None:
                diff_loss = diff_loss * batch_mask
        else:
            diff_loss = torch.zeros([len(predict_output_adj)])
            if self.use_cuda:
                diff_loss = diff_loss.cuda()

        if graph_negative_mask is not None:
            ones = torch.ones([len(predict_output_adj)])
            if self.use_cuda:
                ones = ones.cuda()
            graph_negative_mask = to_pt(graph_negative_mask, self.use_cuda, type='float')  # type='long'
            graph_positive_mask = real_output_adj  # type='long'
            pred_pos_loss = all_loss * graph_positive_mask
            pred_neg_loss = all_loss * graph_negative_mask

            pred_pos_loss_sum = torch.sum(pred_pos_loss, dim=(1, 2, 3))
            pred_pos_loss_num_ = torch.sum(graph_positive_mask, dim=(1, 2, 3))
            pred_pos_loss_num = torch.where(pred_pos_loss_num_ > 0, pred_pos_loss_num_, ones)
            pred_pos_loss = pred_pos_loss_sum / pred_pos_loss_num

            pred_neg_loss_sum = torch.sum(pred_neg_loss, dim=(1, 2, 3))
            pred_neg_loss_num_ = torch.sum(graph_negative_mask, dim=(1, 2, 3))
            pred_neg_loss_num = torch.where(pred_neg_loss_num_ > 0, pred_neg_loss_num_, ones)
            pred_neg_loss = pred_neg_loss_sum / pred_neg_loss_num

            pred_loss = pred_neg_loss + pred_pos_loss
            if batch_mask is not None:
                pred_loss = pred_loss * batch_mask
        else:
            pred_loss = torch.zeros([len(predict_output_adj)])
            if self.use_cuda:
                pred_loss = pred_loss.cuda()

        return pred_loss, diff_loss

    def compute_updated_dynamics(self, input_adj_m, actions, observations,
                                 goal_sentences=None, rewards=None, hx=None, cx=None):
        """update the object dynamics with the dynamics model"""
        if not torch.is_tensor(input_adj_m):
            input_adj = to_pt(input_adj_m, self.use_cuda, type='float')
        else:
            input_adj = input_adj_m
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        node_encodings, relation_encodings, node_embeddings, node_mask = self.model.encode_graph(input_node_name,
                                                                                                 input_relation_name,
                                                                                                 input_adj)
        input_actions = self.get_word_input(actions, minimum_len=10)  # batch x action_len
        action_encodings_sequences, action_mask = self.model.encode_dynamic_action(input_actions)
        if goal_sentences is not None:
            input_goals = self.get_word_input(goal_sentences, minimum_len=20)  # batch x goal_len
            goal_encodings_sequences, goal_mask = self.model.encode_dynamic_goal(input_goals)
        else:
            goal_encodings_sequences, goal_mask = None, None
        input_obs = self.get_word_input(observations, minimum_len=50)
        obs_encoding_sequence, obs_mask = self.model.encode_dynamic_obs(input_obs)

        if 'label' in self.model.dynamic_loss_type:
            predicted_encodings, hx_new, cx_new, attn_mask = self.model.compute_transition_dynamics(
                action_encoding_sequence=action_encodings_sequences,
                obs_encoding_sequence=obs_encoding_sequence,
                action_mask=action_mask,
                obs_mask=obs_mask,
                rewards=rewards,
                goal_encoding_sequence=goal_encodings_sequences,
                goal_mask=goal_mask,
                node_encodings=node_encodings,
                node_embeddings=node_embeddings,
                node_mask=node_mask,
                hx=hx, cx=cx)
            if goal_sentences is not None:
                return predicted_encodings, hx_new, cx_new, node_mask, \
                       input_node_name, input_relation_name, \
                       node_encodings, relation_encodings, \
                       action_encodings_sequences, action_mask, \
                       goal_encodings_sequences, goal_mask
            else:
                return predicted_encodings, hx_new, cx_new, node_mask, \
                       input_node_name, input_relation_name, \
                       node_encodings, relation_encodings, \
                       action_encodings_sequences, action_mask

        elif 'latent' in self.model.dynamic_loss_type:
            predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
            predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, attn_mask = \
                self.model.compute_transition_dynamics(
                    action_encoding_sequence=action_encodings_sequences,
                    obs_encoding_sequence=obs_encoding_sequence,
                    action_mask=action_mask,
                    obs_mask=obs_mask,
                    rewards=rewards,
                    goal_encoding_sequence=goal_encodings_sequences,
                    goal_mask=goal_mask,
                    node_encodings=node_encodings,
                    node_embeddings=node_embeddings,
                    node_mask=node_mask,
                    hx=hx, cx=cx,
                )
            if goal_sentences is not None:
                return predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
                       predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
                       input_node_name, input_relation_name, node_encodings, relation_encodings, \
                       action_encodings_sequences, action_mask, \
                       goal_encodings_sequences, goal_mask
            else:
                return predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
                       predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
                       input_node_name, input_relation_name, node_encodings, relation_encodings, \
                       action_encodings_sequences, action_mask

    def get_graph_node_name_input(self):
        res = copy.copy(self.node_vocab)
        input_node_name = self.get_word_input(res)  # num_node x words
        return input_node_name

    def get_graph_relation_name_input(self):
        res = copy.copy(self.relation_vocab)
        res = [item.replace("_", " ") for item in res]
        input_relation_name = self.get_word_input(res)  # num_node x words
        return input_relation_name

    def get_word_input(self, input_strings, minimum_len=1):
        word_list = [item.split() for item in input_strings]
        word_id_list = [_words_to_ids(tokens, self.word2id) for tokens in word_list]
        maxlen = max_len(word_id_list) if max_len(word_id_list) > minimum_len else minimum_len
        input_word = pad_sequences(word_id_list, maxlen=maxlen).astype('int32')
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def dynamic_latent_loss(self, actions, observations, input_adj_m,
                            goal_sentences=None, rewards=None, hx=None, cx=None, if_clamp_latent_loss=False):
        if goal_sentences is None:
            predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
            predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
            input_node_name, input_relation_name, node_encodings, relation_encodings, \
            action_encodings_sequences, action_mask = \
                self.compute_updated_dynamics(input_adj_m=input_adj_m,
                                              actions=actions,
                                              observations=observations,
                                              goal_sentences=goal_sentences,
                                              rewards=rewards,
                                              hx=hx,
                                              cx=cx, )
        else:
            predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
            predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, node_mask, \
            input_node_name, input_relation_name, node_encodings, relation_encodings, \
            action_encodings_sequences, action_mask, goal_encodings_sequences, goal_mask = \
                self.compute_updated_dynamics(input_adj_m=input_adj_m,
                                              actions=actions,
                                              observations=observations,
                                              goal_sentences=goal_sentences,
                                              rewards=rewards,
                                              hx=hx,
                                              cx=cx, )
        node_latent_loss_all = []
        for i in range(len(self.node_vocab)):
            node_latent_loss = kl_divergence(mu_q=hx_new_mu_post[:, i, :],
                                             logvar_q=hx_new_logvar_post[:, i, :],
                                             mu_p=hx_new_mu_prior[:, i, :],
                                             logvar_p=hx_new_logvar_prior[:, i, :], )
            node_latent_loss_all.append(node_latent_loss)
        latent_loss = torch.stack(node_latent_loss_all, dim=1)

        if goal_sentences is None:
            return torch.mean(latent_loss, dim=1), \
                   predicted_encodings_post, predicted_encodings_prior, relation_encodings, node_mask, \
                   action_encodings_sequences, action_mask
        else:
            return torch.mean(latent_loss, dim=1), \
                   predicted_encodings_post, predicted_encodings_prior, relation_encodings, node_mask, \
                   action_encodings_sequences, action_mask, goal_encodings_sequences, goal_mask

    def save_model_to_path(self, save_to_path,
                           episode_no, eval_acc, eval_loss,
                           log_file, running_game_points={},
                           split_optimizer=False):
        """saving model"""
        if 'rl_planning' in self.task:
            torch.save({
                'epoch': episode_no,
                'model_state_dict': self.model.state_dict(),
                'DQN_optimizer_state_dict': self.optimizers['DQN'].state_dict(),
                'Reward_optimizer_state_dict': self.optimizers['reward_model'].state_dict(),
                'Transition_optimizer_state_dict': self.optimizers['transition_model'].state_dict(),
                'eval_acc': eval_acc,
                'eval_loss': eval_loss,
                'running_game_points': running_game_points,
            }, save_to_path)
        elif 'unsupervised' in self.task and 'goal' in self.task and split_optimizer:
            torch.save({
                'epoch': episode_no,
                'model_state_dict': self.model.state_dict(),
                'Reward_optimizer_state_dict': self.optimizers['reward_model'].state_dict(),
                'Transition_optimizer_state_dict': self.optimizers['transition_model'].state_dict(),
                'eval_acc': eval_acc,
                'eval_loss': eval_loss,
            }, save_to_path)
        else:
            torch.save({
                'epoch': episode_no,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'eval_acc': eval_acc,
                'eval_loss': eval_loss,
            }, save_to_path)
        print("Saved checkpoint to {0} with eval acc {1} and eval loss {2}.".
              format(save_to_path, eval_acc, eval_loss), file=log_file, flush=True)

    def update_target_net(self):
        if self.model.q_target_net is not None:
            self.model.q_target_net.load_state_dict(self.model.q_online_net.state_dict())

    def select_additional_infos_lite(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = False
        request_infos.location = False
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def select_additional_infos(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = True
        request_infos.location = True
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def get_game_info_at_certain_step_lite_batch(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        return observation_strings, action_candidate_list

    def update_dqn(self, candidate_list, action_indices, goal_sentence_list,
                   predicted_node_encodings, predicted_node_mask,
                   rewards,
                   next_candidate_list, next_goal_sentence_list,
                   next_predicted_node_encodings, next_predicted_node_mask,
                   actual_indices, actual_ns, prior_weights):

        dqn_loss, q_value = self.get_dqn_loss(candidate_list=candidate_list,
                                              action_indices=action_indices,
                                              goal_sentence_list=goal_sentence_list,
                                              predicted_node_encodings=predicted_node_encodings,
                                              predicted_node_mask=predicted_node_mask,
                                              rewards=rewards,
                                              next_candidate_list=next_candidate_list,
                                              next_goal_sentence_list=next_goal_sentence_list,
                                              next_predicted_node_encodings=next_predicted_node_encodings,
                                              next_predicted_node_mask=next_predicted_node_mask,
                                              actual_indices=actual_indices,
                                              actual_ns=actual_ns,
                                              prior_weights=prior_weights)
        if dqn_loss is not None:
            # Backpropagate
            self.model.zero_grad()
            self.optimizers['DQN'].zero_grad()
            dqn_loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.q_online_net.parameters(), self.clip_grad_norm)
            self.optimizers['DQN'].step()  # apply gradients
            return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))
        else:
            return None, None

    def sample_dqn_memory(self, episode_no):
        if len(self.dqn_memory.storage) < self.replay_batch_size:
            return None, None, None, None, None, None, None, None, None, None, None, None, None

        data = self.dqn_memory.sample(self.replay_batch_size,
                                      beta=self.beta_scheduler.value(episode_no),
                                      multi_step=self.multi_step)

        if data is None:
            return None, None, None, None, None, None, None, None, None, None, None, None, None

        # if 'unsupervised' in self.task:
        obs_list, pre_action_list, goal_sentence_list, candidate_list, action_indices, \
        predicted_encodings, predicted_masks, rewards, \
        next_obs_list, next_pre_action_list, next_goal_sentence_list, next_candidate_list, \
        next_predicted_encodings, next_predicted_masks, \
        actual_indices, actual_ns, prior_weights = data

        predicted_node_encodings = to_pt(np.asarray(predicted_encodings), enable_cuda=self.use_cuda, type='float')
        predicted_node_mask = to_pt(np.asarray(predicted_masks), enable_cuda=self.use_cuda, type='long')
        next_predicted_node_encodings = to_pt(np.asarray(next_predicted_encodings), enable_cuda=self.use_cuda,
                                              type='float')
        next_predicted_node_mask = to_pt(np.asarray(next_predicted_masks), enable_cuda=self.use_cuda, type='long')

        return candidate_list, action_indices, goal_sentence_list, predicted_node_encodings, predicted_node_mask, rewards, \
               next_candidate_list, next_goal_sentence_list, next_predicted_node_encodings, next_predicted_node_mask, \
               actual_indices, actual_ns, prior_weights

    def update_dqn_dyna(self, episode_no, filter_mask=None, round=1):
        dqn_loss_all = []
        q_value_all = []
        candidate_list, action_indices, goal_sentence_list, predicted_node_encodings, predicted_node_mask, rewards, \
        next_candidate_list, next_goal_sentence_list, next_predicted_node_encodings, next_predicted_node_mask, \
        actual_indices, actual_ns, prior_weights = self.sample_dqn_memory(episode_no)
        if candidate_list is None:
            return None, None

        dqn_loss, q_value = self.update_dqn(candidate_list=candidate_list,
                                            action_indices=action_indices,
                                            goal_sentence_list=goal_sentence_list,
                                            predicted_node_encodings=predicted_node_encodings,
                                            predicted_node_mask=predicted_node_mask,
                                            rewards=rewards,
                                            next_candidate_list=next_candidate_list,
                                            next_goal_sentence_list=next_goal_sentence_list,
                                            next_predicted_node_encodings=next_predicted_node_encodings,
                                            next_predicted_node_mask=next_predicted_node_mask,
                                            actual_indices=actual_indices,
                                            actual_ns=actual_ns,
                                            prior_weights=prior_weights)
        if dqn_loss is not None and q_value is not None:
            dqn_loss_all.append(dqn_loss)
            q_value_all.append(q_value)

        for i in range(round):
            next_predicted_node_encodings, next_predicted_node_mask, random_action_indices, predicted_rewards = \
                self.expand_dqn_memory(predicted_node_encodings=predicted_node_encodings,
                                       predicted_node_mask=predicted_node_mask,
                                       candidate_list=candidate_list,
                                       goal_sentence_list=goal_sentence_list,
                                       filter_mask=filter_mask)
            # we assume the next goal and candidate actions have not been changed
            dqn_loss, q_value = self.update_dqn(candidate_list=candidate_list,
                                                action_indices=random_action_indices,
                                                goal_sentence_list=goal_sentence_list,
                                                predicted_node_encodings=predicted_node_encodings,
                                                predicted_node_mask=predicted_node_mask,
                                                rewards=predicted_rewards,
                                                next_candidate_list=next_candidate_list,
                                                next_goal_sentence_list=next_goal_sentence_list,
                                                next_predicted_node_encodings=next_predicted_node_encodings,
                                                next_predicted_node_mask=next_predicted_node_mask,
                                                actual_indices=actual_indices,
                                                actual_ns=actual_ns,
                                                prior_weights=prior_weights)
            if dqn_loss is not None and q_value is not None:
                dqn_loss_all.append(dqn_loss)
                q_value_all.append(q_value)

        # return to_np(torch.mean(torch.stack(dqn_loss_all))), \
        #        to_np(torch.mean(torch.stack(q_value_all)))
        if len(dqn_loss_all) > 0:
            return np.mean(dqn_loss_all), np.mean(q_value_all)
        else:
            return None, None

    def expand_dqn_memory(self, predicted_node_encodings, predicted_node_mask,
                          candidate_list, goal_sentence_list, filter_mask):
        """
        Apply the model-based rl to expand the dqn memory.
        """
        expand_batch_size = len(candidate_list)
        pad_observations = ['<pad>' for bid in range(expand_batch_size)]
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        # node_embeddings = self.model.get_graph_embed_representations(
        #     input_node_name, type='node')  # 1 x num_node x emb+emb
        relation_embeddings = self.model.get_graph_embed_representations(
            input_relation_name, type='relation')  # 1 x num_relation x emb+emb
        relation_embeddings = relation_embeddings.repeat(expand_batch_size, 1, 1)  # batch x num_relation x emb+emb
        pred_planning_adj = self.model.decode_graph(predicted_node_encodings, relation_embeddings)
        if 'unsupervised' in self.task:
            pred_planner_adj_matrix = pred_planning_adj
        else:
            filter_mask = np.repeat(filter_mask, len(pred_planning_adj), axis=0)
            adj_matrix = (to_np(pred_planning_adj) > 0.5).astype(int)
            pred_planner_adj_matrix = filter_mask * adj_matrix
            pred_planner_adj_matrix = to_pt(pred_planner_adj_matrix, self.use_cuda, type='float')
        node_encodings, relation_encodings, node_embeddings, node_mask = \
            self.model.encode_graph(input_node_name,
                                    input_relation_name,
                                    pred_planner_adj_matrix)
        random_action_indices = [random.randint(0, len(candidate_list[bid]) - 1) for bid in range(expand_batch_size)]
        random_action_list = []
        for bid in range(expand_batch_size):
            random_action_list.append(candidate_list[bid][random_action_indices[bid]])

        input_obs = self.get_word_input(pad_observations, minimum_len=50)
        obs_encoding_sequence, obs_mask = self.model.encode_dynamic_obs(input_obs)

        input_random_actions = self.get_word_input(random_action_list, minimum_len=10)  # batch x action_len
        next_action_encodings_sequences, action_mask = self.model.encode_dynamic_action(input_random_actions)

        if 'unsupervised' in self.task:
            pad_rewards = to_pt(np.asarray([0] * expand_batch_size), enable_cuda=self.use_cuda, type='float')
        else:
            pad_rewards = None
        predicted_next_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
        predicted_next_encodings_post, hx_new_mu_post, hx_new_logvar_post, _ = \
            self.model.compute_transition_dynamics(
                action_encoding_sequence=next_action_encodings_sequences,
                obs_encoding_sequence=obs_encoding_sequence,
                action_mask=action_mask,
                obs_mask=obs_mask,
                rewards=pad_rewards,
                goal_encoding_sequence=None,
                goal_mask=None,
                node_encodings=node_encodings,
                node_embeddings=node_embeddings,
                node_mask=node_mask,
                hx=None, cx=None,
                # post_node_encodings=post_node_encodings,
                # post_node_mask=post_node_mask
            )
        next_predicted_node_encodings = predicted_next_encodings_prior
        next_predicted_node_mask = node_mask
        if 'unsupervised' in self.task:
            input_random_actions = self.get_word_input(random_action_list, minimum_len=10)  # batch x action_len
            next_action_encodings_sequences, action_mask = self.model.encode_text_for_reward_prediction(
                input_random_actions)
            input_goals = self.get_word_input(goal_sentence_list, minimum_len=20)  # batch x goal_len
            goal_encodings_sequences, goal_mask = self.model.encode_text_for_reward_prediction(input_goals)
            pred_rewards = self.compute_rewards_unsupervised(
                predicted_encodings=next_predicted_node_encodings,
                node_mask=next_predicted_node_mask,
                action_encodings_sequences=next_action_encodings_sequences,
                action_mask=action_mask,
                goal_encodings_sequences=goal_encodings_sequences,
                goal_mask=goal_mask)
        else:
            pred_rewards, _ = \
                self.compute_rewards(node_encodings=next_predicted_node_encodings,
                                     node_mask=next_predicted_node_mask,
                                     previous_actions=random_action_list,
                                     goal_sentences=goal_sentence_list,
                                     if_apply_rnn=False,
                                     h_t_minus_one=None,
                                     )
        # if self.use_cuda:
        #     pred_rewards = pred_rewards.detach().cpu().numpy()
        # else:
        #     pred_rewards = pred_rewards.numpy()
        ones = torch.ones([expand_batch_size])
        if self.use_cuda:
            ones = ones.cuda()
        pred_rewards = ones - torch.argmax(pred_rewards, dim=1)  # [pos_prob, neg_prob]

        # if torch.sum(pred_rewards) > 0:
        #     print('Debug')

        return next_predicted_node_encodings, next_predicted_node_mask, random_action_indices, pred_rewards

    def compute_rewards(self, node_encodings, node_mask,
                        # current_observations,
                        previous_actions, goal_sentences,
                        if_apply_rnn, h_t_minus_one, add_softmax=True):

        input_a = self.get_word_input(previous_actions, minimum_len=10)
        action_encoding_sequence, a_mask = self.model.encode_text_for_reward_prediction(input_a)
        h_an = self.model.reward_predict_action_attention(action_encoding_sequence, node_encodings, a_mask, node_mask)
        h_na = self.model.reward_predict_action_attention(node_encodings, action_encoding_sequence, node_mask, a_mask)
        h_an = self.model.reward_predict_action_attention_prj(h_an)  # bs X len X block_hidden_dim
        h_na = self.model.reward_predict_action_attention_prj(h_na)  # bs X len X block_hidden_dim
        ave_h_na = masked_mean(h_na, m=node_mask, dim=1)
        ave_h_an = masked_mean(h_an, m=a_mask, dim=1)

        input_goals = self.get_word_input(goal_sentences, minimum_len=20)
        goal_encoding_sequence, goal_mask = self.model.encode_text_for_reward_prediction(input_goals)
        h_gn = self.model.reward_predict_goal_attention(goal_encoding_sequence, node_encodings, goal_mask, node_mask)
        h_ng = self.model.reward_predict_goal_attention(node_encodings, goal_encoding_sequence, node_mask, goal_mask)
        h_gn = self.model.reward_predict_goal_attention_prj(h_gn)  # bs X len X block_hidden_dim
        h_ng = self.model.reward_predict_goal_attention_prj(h_ng)  # bs X len X block_hidden_dim
        ave_h_ng = masked_mean(h_ng, m=node_mask, dim=1)
        ave_h_gn = masked_mean(h_gn, m=goal_mask, dim=1)

        atten_output = self.model.reward_predict_attention_to_output(
            torch.cat([ave_h_ng, ave_h_gn, ave_h_na, ave_h_an], dim=1))
        atten_output = torch.tanh(atten_output)
        atten_output = self.model.reward_predict_attention_to_output_2(atten_output)
        atten_output = torch.tanh(atten_output)  # batch x block_hidden_dim

        # pred_rewards = self.model.predict_reward_from_node_encodings(node_encodings)
        if if_apply_rnn:
            rnn_output = self.model.reward_pred_graph_rnncell(atten_output, h_t_minus_one) \
                if h_t_minus_one is not None else self.model.reward_pred_graph_rnncell(
                atten_output)  # both batch x block_hidden_dim
            pred_rewards = self.model.reward_linear_predictor1(rnn_output)
        else:
            rnn_output = None
            pred_rewards = self.model.reward_linear_predictor1(atten_output)
        pred_rewards = torch.tanh(pred_rewards)
        pred_rewards = self.model.reward_linear_predictor2(pred_rewards)
        # pred_rewards = pred_rewards.squeeze()

        if add_softmax:
            pred_rewards = F.softmax(pred_rewards, dim=1)

        return pred_rewards, rnn_output

    def compute_rewards_unsupervised(self,
                                     predicted_encodings,
                                     node_mask,
                                     action_encodings_sequences,
                                     action_mask,
                                     goal_encodings_sequences,
                                     goal_mask):
        if self.if_condition_decoder:
            predicted_encodings = predicted_encodings.detach()
            if 'semi' in self.task:
                h_an4r = self.model.reward_predict_action_attention(action_encodings_sequences, predicted_encodings,
                                                                    action_mask, node_mask)
                h_na4r = self.model.reward_predict_action_attention(predicted_encodings, action_encodings_sequences,
                                                                    node_mask, action_mask)
                h_gn4r = self.model.reward_predict_goal_attention(goal_encodings_sequences, predicted_encodings,
                                                                  goal_mask, node_mask)
                h_ng4r = self.model.reward_predict_goal_attention(predicted_encodings, goal_encodings_sequences,
                                                                  node_mask, goal_mask)
            else:
                raise ("Error in unsupervised reward prediction")
            h_an4r = self.model.reward_predict_act_attention_prj(h_an4r)
            h_na4r = self.model.reward_predict_act_attention_prj(h_na4r)
            h_gn4r = self.model.reward_predict_goal_attention_prj(h_gn4r)
            h_ng4r = self.model.reward_predict_goal_attention_prj(h_ng4r)
            ave_h_an4r = masked_mean(h_an4r, m=action_mask, dim=1)
            ave_h_na4r = masked_mean(h_na4r, m=node_mask, dim=1)
            ave_h_gn4r = masked_mean(h_gn4r, m=goal_mask, dim=1)
            ave_h_ng4r = masked_mean(h_ng4r, m=node_mask, dim=1)
            reward_prediction_encoding = torch.cat([ave_h_an4r, ave_h_na4r, ave_h_gn4r, ave_h_ng4r], dim=-1)
            reward_prediction_encoding = self.model.reward_predict_attention_to_output(reward_prediction_encoding)
            reward_prediction_encoding = torch.tanh(reward_prediction_encoding)
            reward_prediction_encoding = self.model.reward_predict_attention_to_output_2(reward_prediction_encoding)
            reward_prediction_encoding = torch.tanh(reward_prediction_encoding)  # batch x block_hidden_dim
        else:
            reward_prediction_encoding = masked_mean(predicted_encodings, m=node_mask, dim=1)

        if self.if_reward_dropout:
            reward_prediction_encoding = self.model.reward_dropout_layer(reward_prediction_encoding)
        pred_rewards = self.model.reward_linear_predictor1(reward_prediction_encoding)
        pred_rewards = torch.tanh(pred_rewards)
        if self.if_reward_dropout:
            pred_rewards = self.model.reward_dropout_layer(pred_rewards)
        pred_rewards = self.model.reward_linear_predictor2(pred_rewards)

        return pred_rewards

    def get_dqn_loss(self, candidate_list, action_indices, goal_sentence_list,
                     predicted_node_encodings, predicted_node_mask,
                     rewards,
                     next_candidate_list, next_goal_sentence_list,
                     next_predicted_node_encodings, next_predicted_node_mask,
                     actual_indices, actual_ns, prior_weights):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        selected_action_list = []
        for bid in range(len(candidate_list)):
            selected_action_list.append(candidate_list[bid][action_indices[bid]])
        input_action_ids = self.get_word_input(selected_action_list, minimum_len=10)  # batch x action_len
        action_encoding_sequence, a_mask = self.model.encode_text_for_q_values(input_action_ids)
        input_goal_ids = self.get_word_input(goal_sentence_list, minimum_len=20)  # batch x action_len
        goal_encoding_sequence, goal_mask = self.model.encode_text_for_q_values(input_goal_ids)
        q_value, _ = self.model.q_online_net(action_encoding_sequence=action_encoding_sequence,
                                             a_mask=a_mask,
                                             goal_encoding_sequence=goal_encoding_sequence,
                                             goal_mask=goal_mask,
                                             node_encodings=predicted_node_encodings,
                                             node_mask=predicted_node_mask)

        with torch.no_grad():
            goal_end_mask = []
            for next_goal_sentence in next_goal_sentence_list:
                if 'END' in next_goal_sentence:
                    goal_end_mask.append(0)
                else:
                    goal_end_mask.append(1)
            goal_end_mask = to_pt(np.asarray(goal_end_mask), enable_cuda=self.use_cuda)
            next_input_goal_ids = self.get_word_input(next_goal_sentence_list, minimum_len=20)  # batch x action_len
            next_input_candidate_word_ids = self.get_action_candidate_list_input(next_candidate_list)
            pred_next_candidate_online_q_values, cand_mask = \
                self.compute_q_values_multi_candidates(node_encodings=next_predicted_node_encodings,
                                                       node_mask=next_predicted_node_mask,
                                                       input_candidate_word_ids=next_input_candidate_word_ids,
                                                       input_goals_ids=next_input_goal_ids,
                                                       model_type='online'
                                                       )
            next_action_indices, _ = self.choose_maxQ_action(action_rank=pred_next_candidate_online_q_values,
                                                             action_mask=cand_mask)
            pred_next_candidate_target_q_values, cand_mask = \
                self.compute_q_values_multi_candidates(node_encodings=next_predicted_node_encodings,
                                                       node_mask=next_predicted_node_mask,
                                                       input_candidate_word_ids=next_input_candidate_word_ids,
                                                       input_goals_ids=next_input_goal_ids,
                                                       model_type='target'
                                                       )

            next_q_value = ez_gather_dim_1(pred_next_candidate_target_q_values, next_action_indices)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda,
                             type="float")

        bellman_score = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, bellman_score, reduce=False)  # batch

        prior_weights = to_pt(prior_weights, enable_cuda=self.use_cuda, type="float")
        loss = loss * prior_weights
        loss = torch.mean(loss)

        abs_td_error = np.abs(to_np(q_value - bellman_score))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        self.dqn_memory.update_priorities(actual_indices, new_priorities)
        return loss, q_value

    def get_action_candidate_list_input(self, action_candidate_list):
        # action_candidate_list (list): batch x num_candidate of strings
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        input_action_candidate_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(action_candidate_list[i], minimum_len=10)
            input_action_candidate_list.append(word_level)
        max_word_num = max([item.size(1) for item in input_action_candidate_list])

        input_action_candidate = np.zeros((batch_size, max_num_candidate, max_word_num))
        input_action_candidate = to_pt(input_action_candidate, self.use_cuda, type="long")
        for i in range(batch_size):
            input_action_candidate[i, :input_action_candidate_list[i].size(0),
            :input_action_candidate_list[i].size(1)] = input_action_candidate_list[i]
        return input_action_candidate

    def choose_maxQ_action(self, action_rank, action_mask=None, if_return_tensor=True):
        """
        Generate an action by maximum q values.
        """
        action_rank_pos = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2
        # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank_pos.size(), (action_mask.size().shape, action_rank_pos.size())
            action_rank_pos = action_rank_pos * action_mask
        action_indices = torch.argmax(action_rank_pos, -1)  # batch
        if if_return_tensor:
            return action_indices, torch.stack([action_rank[i][action_indices[i]]
                                                for i in range(len(action_rank_pos))], dim=0)
        else:
            action_rank_return = []
            for i in range(len(action_rank_pos)):
                if self.use_cuda:
                    action_rank_return.append(action_rank[i][action_indices[i]].data.cpu().item())
                else:
                    action_rank_return.append(action_rank[i][action_indices[i]].data.item())
            return to_np(action_indices), np.asarray(action_rank_return)

    def compute_q_values_multi_candidates(self, node_encodings, node_mask,
                                          input_candidate_word_ids,
                                          input_goals_ids, model_type='online'):
        with torch.no_grad():
            batch_size, num_candidate, candidate_len = input_candidate_word_ids.size(0), \
                                                       input_candidate_word_ids.size(1), \
                                                       input_candidate_word_ids.size(2)
            input_candidate_word_ids = input_candidate_word_ids.view(batch_size * num_candidate, candidate_len)
            # tmp = node_encodings.repeat(num_candidate, 1)

            node_encodings_expanded = torch.stack([node_encodings] * num_candidate, 1).view(
                batch_size * num_candidate,
                node_encodings.size(-2),
                node_encodings.size(-1)
            )
            node_mask_expanded = torch.stack([node_mask] * num_candidate, 1).view(
                batch_size * num_candidate,
                node_mask.size(-1)
            )
            input_goals_expanded = torch.stack([input_goals_ids] * num_candidate, 1).view(
                batch_size * num_candidate,
                input_goals_ids.size(-1)
            )

            input_a = input_candidate_word_ids
            action_encoding_sequence, a_mask = self.model.encode_text_for_q_values(input_a)
            input_goals = input_goals_expanded
            goal_encoding_sequence, goal_mask = self.model.encode_text_for_q_values(input_goals)

            if model_type == 'online':
                pred_q_values, cand_mask = self.model.q_online_net(action_encoding_sequence=action_encoding_sequence,
                                                                   a_mask=a_mask,
                                                                   goal_encoding_sequence=goal_encoding_sequence,
                                                                   goal_mask=goal_mask,
                                                                   node_encodings=node_encodings_expanded,
                                                                   node_mask=node_mask_expanded)
            elif model_type == 'target':
                pred_q_values, cand_mask = self.model.q_target_net(action_encoding_sequence=action_encoding_sequence,
                                                                   a_mask=a_mask,
                                                                   goal_encoding_sequence=goal_encoding_sequence,
                                                                   goal_mask=goal_mask,
                                                                   node_encodings=node_encodings_expanded,
                                                                   node_mask=node_mask_expanded)
            else:
                raise ValueError("Unknown model type {0}".format(model_type))

            cand_mask = cand_mask.view(batch_size, num_candidate, candidate_len)
            cand_mask = cand_mask.byte().any(-1).float()  # batch x num_candidate
            pred_q_values = pred_q_values.view(batch_size, num_candidate)
            pred_q_values = pred_q_values * cand_mask

        return pred_q_values, cand_mask

    def get_unsupervised_dynamics_logistic(self,
                                           input_adj_m,
                                           actions,
                                           observations,
                                           real_rewards,
                                           goal_sentences=None,
                                           batch_masks=None,
                                           goal_extraction_targets=None,
                                           goal_extraction_masks=None,
                                           hx=None,
                                           cx=None,
                                           loss_type='BCE',
                                           if_loss_mean=True,
                                           decode_mode='recon'):

        input_observation_strings = [" ".join(["<bos>"] + item.split()) for item in observations]
        output_observation_strings = [" ".join(item.split() + ["<eos>"]) for item in observations]
        if goal_extraction_targets is not None:
            input_goal_strings = [" ".join(["<bos>"] + item.split()) for item in goal_extraction_targets]
            output_goal_strings = [" ".join(item.split() + ["<eos>"]) for item in goal_extraction_targets]
        rewards = to_pt(np.asarray(real_rewards), enable_cuda=self.use_cuda, type='float')

        if 'latent' in self.model.dynamic_loss_type:
            if "goal" not in self.model.dynamic_loss_type:
                latent_loss, predicted_encodings_post, predicted_encodings_prior, relation_encodings, node_mask, \
                action_encodings_sequences, action_mask = \
                    self.dynamic_latent_loss(actions=actions,
                                             observations=observations,
                                             input_adj_m=input_adj_m,
                                             goal_sentences=None,
                                             rewards=rewards,
                                             hx=hx,
                                             cx=cx,
                                             )
                predict_output_adj = self.model.decode_graph(predicted_encodings_post, relation_encodings)
            else:
                latent_loss, predicted_encodings_post, predicted_encodings_prior, relation_encodings, node_mask, \
                _, _, _, _ = \
                    self.dynamic_latent_loss(actions=actions,
                                             observations=observations,
                                             input_adj_m=input_adj_m,
                                             goal_sentences=goal_sentences,
                                             rewards=rewards,
                                             hx=hx,
                                             cx=cx,
                                             )
                predict_output_adj = self.model.decode_graph(predicted_encodings_post, relation_encodings)

            predict_output_adj = predict_output_adj.detach()
            latent_loss = latent_loss * batch_masks
            if decode_mode == 'recon':  # reconstruction
                predicted_encodings = predicted_encodings_post
            elif decode_mode == 'pred':  # prediction
                predicted_encodings = predicted_encodings_prior
            else:
                raise ValueError("Unknown decode model: {0}".format(decode_mode))
        elif 'label' in self.model.dynamic_loss_type:
            if goal_sentences is None:
                predicted_encodings, hx_new, cx_new, node_mask, \
                input_node_name, input_relation_name, \
                node_encodings_input, relation_encodings, \
                _, _, \
                    = self.compute_updated_dynamics(input_adj_m, actions, observations, goal_sentences, rewards, hx, cx)
            else:
                predicted_encodings, hx_new, cx_new, node_mask, \
                input_node_name, input_relation_name, \
                node_encodings_input, relation_encodings, \
                _, _, \
                _, _ \
                    = self.compute_updated_dynamics(input_adj_m, actions, observations, goal_sentences, rewards, hx, cx)
            predict_output_adj = self.model.decode_graph(predicted_encodings, relation_encodings)
        else:
            raise ValueError("Unknown loss type: {0}".format(self.model.dynamic_loss_type))

        obs_gen_loss, obs_gen_pred, obs_gen_target_mask = \
            self.dynamic_obs_gen_label_loss(input_observation_strings=input_observation_strings,
                                            output_observation_strings=output_observation_strings,
                                            actions_words=actions,
                                            predicted_encodings=predicted_encodings,
                                            node_mask=node_mask,
                                            episode_masks=batch_masks)
        obs_gen_pred = obs_gen_pred * obs_gen_target_mask.unsqueeze(-1)
        if goal_extraction_targets is not None and torch.sum(goal_extraction_masks) > 0:
            goal_gen_loss, goal_gen_pred_tf, goal_gen_target_mask = \
                self.dynamic_goal_gen_label_loss(input_goal_strings=input_goal_strings,
                                                 output_goal_strings=output_goal_strings,
                                                 observation_strings=observations,
                                                 predicted_encodings=predicted_encodings,
                                                 node_mask=node_mask,
                                                 goal_extract_masks=goal_extraction_masks)
            goal_gen_pred_tf = goal_gen_pred_tf * goal_gen_target_mask.unsqueeze(-1)
            goal_gen_pred_greedy_words = self.goal_greedy_generation(observation_strings=observations,
                                                                     predicted_encodings=predicted_encodings,
                                                                     node_mask=node_mask,
                                                                     gen_goal_sentence_len=80 if self.difficulty_level == 9 else 20)
        else:
            goal_gen_pred_tf, goal_gen_pred_greedy_words, goal_gen_target_mask = None, None, None
            goal_gen_loss = torch.tensor(0).double()
            if self.use_cuda:
                goal_gen_loss = goal_gen_loss.cuda()

        reward_loss, pred_rewards_array, real_rewards_array, episode_masks_ = \
            self.dynamic_reward_label_loss(real_rewards=real_rewards,
                                           predicted_encodings=predicted_encodings,
                                           node_mask=node_mask,
                                           episode_masks=batch_masks,
                                           loss_type=loss_type,
                                           action_words=actions,
                                           goal_sentences=goal_sentences,
                                           )

        total_number = torch.sum(episode_masks_) if torch.sum(episode_masks_) > 0 else torch.tensor(1)
        if self.use_cuda:
            total_number = total_number.cuda()
        if 'latent' in self.model.dynamic_loss_type:
            if if_loss_mean:
                return obs_gen_loss, predict_output_adj, obs_gen_pred, \
                       torch.sum(reward_loss) / total_number, pred_rewards_array, real_rewards_array, \
                       torch.sum(latent_loss) / total_number, \
                       goal_gen_loss, goal_gen_pred_tf, goal_gen_pred_greedy_words
            else:
                return obs_gen_loss, predict_output_adj, obs_gen_pred, \
                       reward_loss, pred_rewards_array, real_rewards_array, \
                       latent_loss, \
                       goal_gen_loss, goal_gen_pred_tf, goal_gen_pred_greedy_words
        elif 'label' in self.model.dynamic_loss_type:
            if if_loss_mean:
                return obs_gen_loss, predict_output_adj, obs_gen_pred, \
                       torch.sum(reward_loss) / total_number, pred_rewards_array, real_rewards_array, \
                       torch.mean(to_pt(np.zeros([len(predict_output_adj)]), self.use_cuda, type='float')), \
                       goal_gen_loss, goal_gen_pred_tf, goal_gen_pred_greedy_words
            else:
                return obs_gen_loss, predict_output_adj, obs_gen_pred, \
                       reward_loss, pred_rewards_array, real_rewards_array, \
                       to_pt(np.zeros([len(predict_output_adj)]), self.use_cuda, type='float'), \
                       goal_gen_loss, goal_gen_pred_tf, goal_gen_pred_greedy_words
        else:
            raise ValueError("Unknown loss type: {0}".format(self.model.dynamic_loss_type))

    def dynamic_obs_gen_label_loss(self, input_observation_strings, output_observation_strings,
                                   actions_words, predicted_encodings, node_mask, episode_masks):
        input_actions = self.get_word_input(actions_words, minimum_len=10)  # batch x action_len
        action_encodings_sequences, action_mask = self.model.encode_text_for_obs_prediction(input_actions)
        h_an4o = self.model.obs_gen_action_attention(action_encodings_sequences, predicted_encodings,
                                                     action_mask, node_mask)
        h_na4o = self.model.obs_gen_action_attention(predicted_encodings, action_encodings_sequences,
                                                     node_mask, action_mask)
        h_an4o = self.model.obs_gen_attention_prj(h_an4o)
        h_na4o = self.model.obs_gen_attention_prj(h_na4o)

        input_targets = self.get_word_input(input_observation_strings)
        ground_truths = self.get_word_input(output_observation_strings)
        target_mask = compute_mask(input_targets)
        pred, _ = self.model.decode_for_word_gen(input_target_word_ids=input_targets,
                                                 h_item_n=h_an4o,
                                                 word_mask=action_mask,
                                                 h_n_item=h_na4o,
                                                 node_mask=node_mask,
                                                 input_words=input_actions,
                                                 gen_target='obs')
        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truths, target_mask,
                                     smoothing_eps=self.smoothing_eps)
        obs_gen_loss = torch.sum(batch_loss * episode_masks) / torch.sum(episode_masks)
        # only place where using `episodes_masks`
        return obs_gen_loss, pred, target_mask

    def dynamic_goal_gen_label_loss(self, input_goal_strings, output_goal_strings,
                                    observation_strings, predicted_encodings, node_mask, goal_extract_masks):
        predicted_encodings = predicted_encodings.detach()
        input_obss = self.get_word_input(observation_strings, minimum_len=50)  # batch x action_len
        obs_encodings_sequences, obs_mask = self.model.encode_text_for_goal_prediction(input_obss)
        h_on4g = self.model.goal_gen_obs_attention(obs_encodings_sequences, predicted_encodings,
                                                   obs_mask, node_mask)
        h_no4g = self.model.goal_gen_obs_attention(predicted_encodings, obs_encodings_sequences,
                                                   node_mask, obs_mask)
        h_on4g = self.model.goal_gen_attention_prj(h_on4g)
        h_no4g = self.model.goal_gen_attention_prj(h_no4g)

        input_targets = self.get_word_input(input_goal_strings)
        ground_truths = self.get_word_input(output_goal_strings)
        target_mask = compute_mask(input_targets)
        pred, _ = self.model.decode_for_word_gen(input_target_word_ids=input_targets,
                                                 h_item_n=h_on4g,
                                                 word_mask=obs_mask,
                                                 h_n_item=h_no4g,
                                                 node_mask=node_mask,
                                                 input_words=input_obss,
                                                 gen_target='goal')
        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truths, target_mask,
                                     smoothing_eps=self.smoothing_eps)
        goal_extract_masks_sum = torch.sum(goal_extract_masks) if torch.sum(goal_extract_masks) > 0 else 1
        goal_gen_loss = torch.sum(batch_loss * goal_extract_masks) / goal_extract_masks_sum
        return goal_gen_loss, pred, target_mask

    def goal_greedy_generation(self, observation_strings, predicted_encodings, node_mask,
                               gen_goal_sentence_len):
        with torch.no_grad():
            batch_size = len(observation_strings)

            input_obss = self.get_word_input(observation_strings, minimum_len=50)  # batch x action_len
            obs_encodings_sequences, obs_mask = self.model.encode_text_for_goal_prediction(input_obss)
            h_on4g = self.model.goal_gen_obs_attention(obs_encodings_sequences, predicted_encodings,
                                                       obs_mask, node_mask)
            h_no4g = self.model.goal_gen_obs_attention(predicted_encodings, obs_encodings_sequences,
                                                       node_mask, obs_mask)
            h_on4g = self.model.goal_gen_attention_prj(h_on4g)
            h_no4g = self.model.goal_gen_attention_prj(h_no4g)

            input_target_token_list = [["<bos>"] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(gen_goal_sentence_len):
                input_targets = self.get_word_input([" ".join(item) for item in input_target_token_list])
                pred, _ = self.model.decode_for_word_gen(input_target_word_ids=input_targets,
                                                         h_item_n=h_on4g,
                                                         word_mask=obs_mask,
                                                         h_n_item=h_no4g,
                                                         node_mask=node_mask,
                                                         input_words=input_obss,
                                                         gen_target='goal')
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [self.word_vocab[pred[b]]] if eos[b] == 0 else []
                    input_target_token_list[b] = input_target_token_list[b] + new_stuff
                    if pred[b] == self.word2id["<eos>"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            return [" ".join(item[1:]) for item in input_target_token_list]

    def dynamic_reward_label_loss(self, real_rewards, predicted_encodings, node_mask, episode_masks, loss_type,
                                  action_words, goal_sentences):
        input_actions = self.get_word_input(action_words, minimum_len=10)  # batch x action_len
        action_encodings_sequences, action_mask = self.model.encode_text_for_reward_prediction(input_actions)
        if goal_sentences is not None:
            input_goals = self.get_word_input(goal_sentences, minimum_len=20)  # batch x goal_len
            goal_encodings_sequences, goal_mask = self.model.encode_text_for_reward_prediction(input_goals)
        else:
            goal_encodings_sequences, goal_mask = None, None
        pred_rewards = self.compute_rewards_unsupervised(
            predicted_encodings=predicted_encodings,
            node_mask=node_mask,
            action_encodings_sequences=action_encodings_sequences,
            action_mask=action_mask,
            goal_encodings_sequences=goal_encodings_sequences,
            goal_mask=goal_mask)

        real_rewards_binary = []
        for real_reward in real_rewards:
            real_rewards_binary.append([real_reward, 1 - real_reward])
        real_rewards_binary = to_pt(np.asarray(real_rewards_binary), self.use_cuda, type='float')

        episode_masks_ = episode_masks.unsqueeze(1).repeat(1, 2)
        # dump_labels = torch.stack([torch.ones([len(pred_rewards)]), torch.zeros([len(pred_rewards)])], dim=1)
        dump_labels = torch.ones([len(pred_rewards), 2]) * 0.5
        if self.use_cuda:
            dump_labels = dump_labels.cuda()
        pred_rewards = torch.where(episode_masks_ > 0, pred_rewards, dump_labels)
        if loss_type == 'BCE':
            # reward_loss = self.model.bce_logits_loss(input=pred_rewards, target=real_rewards_binary)
            pred_rewards = F.softmax(pred_rewards, dim=1)
            # softmax = torch.nn.Softmax(dim=1)
            # pred_rewards = softmax(pred_rewards)
            # pred_rewards = torch.clamp(pred_rewards, min=0, max=1)
            # assert torch.max(pred_rewards) <= 1
            # assert torch.min(pred_rewards) >= 0
            reward_loss = self.model.bce_loss(input=pred_rewards, target=real_rewards_binary)
        elif loss_type == 'L1Smooth':
            reward_loss = self.model.smooth_l1_loss(input=pred_rewards, target=real_rewards_binary)
        else:
            raise ValueError("Wrong loss type {0}".format(loss_type))

        reward_loss = reward_loss * episode_masks_  # only place where using `episodes_masks`
        pred_rewards_array = pred_rewards.cpu().detach().numpy()
        real_rewards_array = real_rewards_binary.cpu().detach().numpy()

        return reward_loss, pred_rewards_array, real_rewards_array, episode_masks_

    def update_rl_models(self, poss_triplets_mask, episode_no, log_file):
        self.train()
        reward_loss, transition_pred_loss, transition_diff_loss, transition_latent_loss = \
            self.get_rl_model_loss(poss_triplets_mask, episode_no, log_file)

        if reward_loss is not None:
            # Backpropagate
            self.model.zero_grad()
            self.optimizers['reward_model'].zero_grad()
            reward_loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizers['reward_model'].step()  # apply gradients

            parameters_info = []
            for k, v in self.model.named_parameters():
                if v.grad is not None:
                    parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                else:
                    parameters_info.append("{0}:{1}".format(k, v.grad))
            print(parameters_info, file=log_file, flush=True)

            # Backpropagate
            self.model.zero_grad()
            self.optimizers['transition_model'].zero_grad()
            transition_loss = transition_pred_loss + transition_diff_loss + transition_latent_loss
            transition_loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizers['transition_model'].step()  # apply gradients

            parameters_info = []
            for k, v in self.model.named_parameters():
                if v.grad is not None:
                    parameters_info.append("{0}:{1}".format(k, torch.mean(v.grad)))
                else:
                    parameters_info.append("{0}:{1}".format(k, v.grad))
            print(parameters_info, file=log_file, flush=True)

    def get_rl_model_loss(self, poss_triplets_mask, episode_no, log_file):

        # print(len(self.model_memory.storage[0]))
        # print(len(self.model_memory.storage[1]))

        if len(self.model_memory.storage[0]) < int(self.replay_batch_size / 2) or \
                len(self.model_memory.storage[1]) < int(self.replay_batch_size / 2):
            return None, None, None, None

        data = self.model_memory.sample(self.replay_batch_size)

        if data is None:
            return None, None, None, None

        [observations, graph_triplets, goals, selected_action,
         next_observations, next_graph_triplets,
         next_rewards] = data

        graph_triplets_adjs = self.get_graph_adjacency_matrix(graph_triplets)
        next_graph_triplets_adjs = self.get_graph_adjacency_matrix(next_graph_triplets)

        curr_load_data_batch_size = len(observations)
        reward_loss, pred_rewards, real_rewards, correct_count, _ = \
            self.reward_prediction_dynamic(previous_adjacency_matrix=graph_triplets_adjs,
                                           real_rewards=[reward for reward in next_rewards],
                                           current_observations=next_observations,
                                           previous_actions=selected_action,
                                           current_goal_sentences=goals,
                                           if_apply_rnn=self.model.reward_predictor_apply_rnn,
                                           )

        graph_diff_pos_sample_mask, graph_diff_neg_sample_mask, diff_triplet_index, \
        input_adjacency_matrix, output_adjacency_matrix, actions, pos_mask, neg_mask = \
            self.get_diff_sample_masks(prev_adjs=graph_triplets_adjs,
                                       target_adjs=next_graph_triplets_adjs,
                                       actions=selected_action
                                       )
        filter_mask_batch = np.repeat(poss_triplets_mask, curr_load_data_batch_size, axis=0)
        graph_negative_mask_agg = filter_mask_batch - output_adjacency_matrix
        tmp_min = np.min(graph_negative_mask_agg)
        assert tmp_min == 0
        tmp_max = np.max(graph_negative_mask_agg)
        assert tmp_max == 1

        transition_pre_loss, transition_diff_loss, transition_latent_loss, _, _, _, _ = self.get_predict_dynamics_logits(
            input_adj_m=input_adjacency_matrix,
            actions=actions,
            observations=next_observations,
            output_adj_m=output_adjacency_matrix,
            hx=None, cx=None,
            graph_diff_pos_sample_mask=graph_diff_pos_sample_mask,
            graph_diff_neg_sample_mask=graph_diff_neg_sample_mask,
            graph_negative_mask=graph_negative_mask_agg,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
        )

        return reward_loss, transition_pre_loss, transition_diff_loss, transition_latent_loss

    def reward_prediction_dynamic(self, previous_adjacency_matrix, real_rewards,
                                  current_observations, previous_actions, current_goal_sentences,
                                  if_apply_rnn,
                                  loss_type='BCE',
                                  current_adjacency_matrix=None,
                                  h_t_minus_one=None,
                                  episode_masks=None,
                                  if_loss_mean=True):

        if 'random' in self.task or 'gtf' in self.task:  # comparsion group
            if 'random' in self.task:
                current_adjacency_matrix = np.random.uniform(low=0.0, high=1.0, size=previous_adjacency_matrix.shape)
            input_adj = to_pt(current_adjacency_matrix, self.use_cuda, type='float')
            input_node_name = self.get_graph_node_name_input()
            input_relation_name = self.get_graph_relation_name_input()
            predicted_encodings, relation_encodings, node_embeddings, node_mask = \
                self.model.encode_graph(input_node_name,
                                        input_relation_name,
                                        input_adj)
        else:
            if self.model.dynamic_loss_type == 'label':
                predicted_encodings, _, _, node_mask, _, _, _, _, _, _ = \
                    self.compute_updated_dynamics(input_adj_m=previous_adjacency_matrix,
                                                  actions=previous_actions,
                                                  observations=current_observations,
                                                  hx=None,
                                                  cx=None)
            elif self.model.dynamic_loss_type == 'latent':
                predicted_encodings, _, _, _, _, _, node_mask, _, _, _, _, _, _ = \
                    self.compute_updated_dynamics(input_adj_m=previous_adjacency_matrix,
                                                  actions=previous_actions,
                                                  observations=current_observations,
                                                  hx=None,
                                                  cx=None)
            else:
                raise ValueError('Unknown model loss type {0}'.format(self.model.dynamic_loss_type))

        pred_rewards, rnn_output = self.compute_rewards(node_encodings=predicted_encodings,
                                                        node_mask=node_mask,
                                                        # current_observations=current_observations,
                                                        previous_actions=previous_actions,
                                                        goal_sentences=current_goal_sentences,
                                                        if_apply_rnn=if_apply_rnn,
                                                        h_t_minus_one=h_t_minus_one,
                                                        add_softmax=True if loss_type == 'BCE' else False
                                                        )
        real_rewards_binary = []
        for real_reward in real_rewards:
            if torch.is_tensor(real_reward):
                real_rewards_binary.append(torch.stack([real_reward, 1 - real_reward]))
            else:
                real_rewards_binary.append([real_reward, 1 - real_reward])
        if torch.is_tensor(real_rewards_binary[0]):
            real_rewards_binary = torch.stack(real_rewards_binary)
        else:
            real_rewards_binary = to_pt(np.asarray(real_rewards_binary), self.use_cuda, type='float')
        if loss_type == 'BCE' and not torch.max(pred_rewards) <= 1:
            print("debug")

        # MSE_loss = torch.nn.MSELoss(reduce=False)
        # assert torch.max(pred_rewards) <= 1
        # assert torch.min(pred_rewards) >= 0
        # assert torch.max(real_rewards_binary) <= 1
        # assert torch.min(real_rewards_binary) >= 0
        if if_apply_rnn:
            episode_masks_ = episode_masks.unsqueeze(1).repeat(1, 2)
            # dump_labels = torch.stack([torch.ones([len(pred_rewards)]), torch.zeros([len(pred_rewards)])], dim=1)
            dump_labels = torch.ones([len(pred_rewards), 2]) * 0.5
            if self.use_cuda:
                dump_labels = dump_labels.cuda()
            pred_rewards = torch.where(episode_masks_ > 0, pred_rewards, dump_labels)
            if loss_type == 'BCE':
                loss = F.binary_cross_entropy(input=pred_rewards, target=real_rewards_binary, reduce=False)
            else:
                loss = F.smooth_l1_loss(input=pred_rewards, target=real_rewards_binary, reduce=False)
            loss = loss * episode_masks_  # only place where using `episodes_masks`
        else:
            if loss_type == 'BCE':
                loss = F.binary_cross_entropy(input=pred_rewards, target=real_rewards_binary, reduce=False)
            else:
                loss = F.smooth_l1_loss(input=pred_rewards, target=real_rewards_binary, reduce=False)
            episode_masks_ = None
        pred_rewards_array = pred_rewards.cpu().detach().numpy()
        real_rewards_array = real_rewards_binary.cpu().detach().numpy()
        correct_count = 0
        for idx in range(len(pred_rewards_array)):
            pred_reward_array = pred_rewards_array[idx]
            pred_ = np.argmax(pred_reward_array)
            real_reward_array = real_rewards_array[idx]
            real_ = np.argmax(real_reward_array)
            if real_ == pred_ and not if_apply_rnn:
                correct_count += 1
            elif real_ == pred_ and if_apply_rnn:
                if episode_masks[idx]:
                    correct_count += 1

        if if_loss_mean:
            mean_loss = torch.sum(loss) / torch.sum(episode_masks_) if if_apply_rnn else torch.mean(loss)
            return mean_loss, pred_rewards_array, real_rewards_array, correct_count, rnn_output
        else:
            return loss, pred_rewards_array, real_rewards_array, correct_count, rnn_output

    def act_during_rl_train(self, node_encodings, node_mask, action_candidate_list, input_goals_ids, force_actions,
                            random=False):

        input_candidate_word_ids = self.get_action_candidate_list_input(action_candidate_list)
        pred_q_values, cand_mask = \
            self.compute_q_values_multi_candidates(node_encodings,
                                                   node_mask,
                                                   input_candidate_word_ids,
                                                   input_goals_ids)

        action_indices_maxq, max_action_values = self.choose_maxQ_action(action_rank=pred_q_values,
                                                                         action_mask=cand_mask,
                                                                         if_return_tensor=False)

        action_indices_random, random_action_values = self.choose_random_action(action_rank=pred_q_values,
                                                                                action_unpadded=cand_mask)

        if self.mode == "eval":
            chosen_indices = action_indices_maxq
            action_values = max_action_values
        elif random:
            chosen_indices = action_indices_random
            action_values = random_action_values
        else:
            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(len(node_encodings),))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon
            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            action_values = less_than_epsilon * random_action_values + greater_than_epsilon * max_action_values

        chosen_indices = chosen_indices.astype(int)
        chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
        action_values_dict = {}
        # for bid in range(len(node_encodings)):
        for action_idx in range(len(action_candidate_list[0])):
            action_values_dict.update({action_candidate_list[0][action_idx]: pred_q_values[0][action_idx]})
        for bid in range(len(force_actions)):
            force_index = force_actions[bid]
            if force_index is not None:
                chosen_indices[bid] = force_index
                action_values[bid] = pred_q_values[bid][force_index]

        return chosen_indices, action_values, node_encodings, node_mask

    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            random_indices = np.random.choice(action_space_size, batch_size)
        else:
            random_indices = []
            candidate_length = to_np(torch.sum(action_unpadded, dim=1))
            for j in range(batch_size):
                random_indices.append(np.random.choice(int(candidate_length[j])))
            random_indices = np.array(random_indices)

        action_rank_return = []
        for i in range(len(action_rank)):
            if self.use_cuda:
                action_rank_return.append(action_rank[i][random_indices[i]].data.cpu().item())
            else:
                action_rank_return.append(action_rank[i][random_indices[i]].data.item())
        return random_indices, np.asarray(action_rank_return)

    def finish_of_episode(self, episode_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no \
                % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.unexplore_rate = self.unexplore_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.epsilon = max(self.epsilon, 0.0)
            self.unexplore_rate = max(self.unexplore_rate, 0.0)
            self.max_nb_steps_per_episode_scale = \
                self.max_nb_steps_scheduler.value(episode_no - self.learn_start_from_this_episode)

    def get_graph_autoencoder_logits(self, input_adj_m, real_adj_m, graph_negative_mask, if_loss_mean=True):
        input_adj = to_pt(input_adj_m, self.use_cuda, type='float')  # type='long'
        real_adj = to_pt(real_adj_m, self.use_cuda, type='float')  # type='long'
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        node_encodings, relation_encodings, node_embeddings, node_mask = self.model.encode_graph(input_node_name,
                                                                                                 input_relation_name,
                                                                                                 input_adj)
        # mu = self.model.vgae_mu(node_encoding, relation_encoding, input_adjacency_matrices)
        # self.mu = mu * node_mask.unsqueeze(-1)
        #
        # log_sigma = self.model.vgae_logstd(node_encoding, relation_encoding, input_adjacency_matrices)
        # self.log_sigma = log_sigma * node_mask.unsqueeze(-1)
        #
        # noise = np.random.normal(size=[input_adjacency_matrices.size(0), self.node_vocab_size, self.vgae_hidden_dim])
        # gaussian_noise = to_pt(noise, self.mu.is_cuda)
        # sampled_z = gaussian_noise * torch.exp(self.log_sigma) + self.mu  # batch x num_node x hidden
        predict_adj = self.model.decode_graph(node_encodings, relation_encodings)
        # tmp = to_np(predict_adj)
        all_loss = F.binary_cross_entropy(input=predict_adj, target=real_adj, reduce=False)

        graph_negative_mask = to_pt(graph_negative_mask, self.use_cuda, type='float')  # type='long'
        graph_positive_mask = to_pt(real_adj_m, self.use_cuda, type='float')  # type='long'

        positive_loss = all_loss * graph_positive_mask
        negative_loss = all_loss * graph_negative_mask

        positive_loss = torch.sum(positive_loss, dim=(1, 2, 3)) / torch.sum(graph_positive_mask, dim=(1, 2, 3))
        negative_loss = torch.sum(negative_loss, dim=(1, 2, 3)) / torch.sum(graph_negative_mask, dim=(1, 2, 3))
        loss = negative_loss + positive_loss
        if if_loss_mean:
            return torch.mean(loss), predict_adj, real_adj
        else:
            return loss, predict_adj, real_adj

    def get_goal_list_input(self, goal_list):
        # action_candidate_list (list): batch x num_candidate of strings
        batch_size = len(goal_list)
        max_num_goals = max_len(goal_list)
        input_goal_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(goal_list[i], minimum_len=20)
            input_goal_list.append(word_level)
        max_word_num = max([item.size(1) for item in input_goal_list])

        input_goals = np.zeros((batch_size, max_num_goals, max_word_num))
        input_goals = to_pt(input_goals, self.use_cuda, type="long")
        for i in range(batch_size):
            input_goals[i, :input_goal_list[i].size(0), :input_goal_list[i].size(1)] = input_goal_list[i]
        return input_goals