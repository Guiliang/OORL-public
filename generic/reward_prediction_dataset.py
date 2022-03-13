import json
import os
from os.path import join as pjoin

import torch
from tqdm import tqdm
import gym
import numpy as np
from generic.data_utils import GraphDataset, compute_current_triplets, adj_to_triplets, matching_object_from_obs, \
    diff_triplets
from generic.model_utils import to_np


class RewardPredictionDynamicDataGoal(gym.Env):
    FILENAMES_MAP = {
        "train": "train_all_goal.json",
        "valid": "valid_goal.json",
        "test": "test_goal.json"
    }

    def __init__(self, config, agent, filter_mask, max_data_size=None, log_file=None, seed=None):
        self.filter_mask = filter_mask
        self.rng = None
        self.config = config
        self.all_data_types = [0, 1]
        # Load dataset splits.
        self.data_sizes = {}
        self.train_sizes = {}
        self.valid_sizes = {}
        self.test_sizes = {}
        self.dataset = {}
        max_diff_tri_num_dict = {}
        avg_diff_tri_num_dict = {}
        self.agent = agent
        self.max_data_size = max_data_size
        self.read_config()
        if seed is None:
            self.seed(self.random_seed)
        else:
            self.seed(seed)

        for split in ["train", "valid", "test"]:  # "train", "valid", "test"
            self.dataset[split] = {}
            for data_type in self.all_data_types:
                self.dataset[split].update({data_type: {"truth_previous_triplets": [],
                                                        "pred_previous_triplets": [],
                                                        "truth_current_triplets": [],
                                                        "pred_current_triplets": [],
                                                        "current_rewards": [],
                                                        "previous_action": [],
                                                        "current_observation": [],
                                                        "current_goal_sentences": [],
                                                        "current_goal_sentences_real": []
                                                        }})

            for difficulty_level in self.data_paths_dict.keys():
                max_diff_tri_num, avg_diff_tri_num = self.load_dataset_for_reward_predict(
                    data_path=self.data_paths_dict[difficulty_level],
                    dynamic_file_name=self.dynamic_file_maps[difficulty_level][split],
                    split=split,
                    log_file=log_file)
                if split not in max_diff_tri_num_dict.keys():
                    max_diff_tri_num_dict.update({split: max_diff_tri_num})
                    avg_diff_tri_num_dict.update({split: avg_diff_tri_num})
                else:
                    if max_diff_tri_num> max_diff_tri_num_dict[split]:
                        max_diff_tri_num_dict.update({split: max_diff_tri_num})
                        avg_diff_tri_num_dict.update({split: avg_diff_tri_num})
        print(
            "loaded dataset from {0} with max triplet difference {1} ...".format(self.data_paths_dict,
                                                                                 max_diff_tri_num_dict),
            file=log_file, flush=True)
        print(
            "loaded dataset from {0} with avg triplet difference {1} ...".format(self.data_paths_dict,
                                                                                 avg_diff_tri_num_dict),
            file=log_file, flush=True)

        for data_type in self.all_data_types:
            self.train_sizes.update({data_type: len(self.dataset["train"][data_type]["current_rewards"])})
            self.valid_sizes.update({data_type: len(self.dataset["valid"][data_type]["current_rewards"])})
            self.test_sizes.update({data_type: len(self.dataset["test"][data_type]["current_rewards"])})
        # self.test_size = len(self.dataset["test"]["current_triplets"])
        self.batch_pointer = None
        self.type_idx_pointer = None
        self.batch_size, self.data = None, None
        self.split = "train"

    def load_dataset_for_reward_predict(self, data_path, dynamic_file_name, split, log_file):

        # max_seq_len = 0
        max_diff_tri_num = 0
        total_pred_diff_num = 0
        total_samples = 0
        dynamic_data_path = pjoin(data_path, dynamic_file_name)
        # file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        desc = "Loading {}".format(os.path.basename(dynamic_data_path))
        print(desc, file=log_file, flush=True)
        with open(dynamic_data_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        # for data_type in self.all_data_types:
        self.dataset[split][self.all_data_types[0]]["graph_dataset"] = graph_dataset
        # current_game_name = None
        # pre_step_k = 1
        # pre_step = None
        emphasize_num = 1
        game_count = 0
        prev_game_name = None
        for example in tqdm(data["examples"], desc=desc, file=log_file):
            # print(example["step"])

            if example['game'] != prev_game_name:
                game_count += 1
                prev_game_name = example['game']
            if game_count > self.max_game_count[split]:
                break
            if self.max_data_size is not None:
                if total_samples > self.max_data_size:
                    break
            if " your score has just gone up by one point ." in example["observation"]:
                obs_bf_len = len(example["observation"])
                # print("find you.")
                # print(example["observation"])
                obs = example["observation"].replace(" your score has just gone up by one point .", "")
                obs_af_len = len(obs)
                assert obs_af_len < obs_bf_len
            else:
                obs = example["observation"]

            data_type = example["current_reward"]
            # self.dataset[split][data_type]["truth_previous_triplets"].append(
            #     graph_dataset.decompress(example["truth_previous_triplets"]))
            # self.dataset[split][data_type]["pred_previous_triplets"].append(
            #     graph_dataset.decompress(example["pred_previous_triplets"]))
            # self.dataset[split][data_type]["truth_current_triplets"].append(
            #     graph_dataset.decompress(example["truth_current_triplets"]))
            # self.dataset[split][data_type]["pred_current_triplets"].append(
            #     graph_dataset.decompress(example["pred_current_triplets"]))
            self.dataset[split][data_type]["truth_previous_triplets"].append(example["truth_previous_triplets"])
            self.dataset[split][data_type]["pred_previous_triplets"].append(example["pred_previous_triplets"])
            self.dataset[split][data_type]["truth_current_triplets"].append(example["truth_current_triplets"])
            self.dataset[split][data_type]["pred_current_triplets"].append(example["pred_current_triplets"])
            self.dataset[split][data_type]["current_rewards"].append(example["current_reward"])
            self.dataset[split][data_type]["previous_action"].append(example["previous_action"])
            self.dataset[split][data_type]["current_observation"].append(obs)
            self.dataset[split][data_type]["current_goal_sentences"].append(example["current_goal_sentence"])
            self.dataset[split][data_type]["current_goal_sentences_real"].append(example["current_goal_sentence_real"])
            # self.dataset[split][data_type]["current_goal_tuples"].append(example["current_goal_tuples"])
            total_samples += 1

        avg_diff_tri_num = float(total_pred_diff_num) / total_samples
        return max_diff_tri_num, avg_diff_tri_num

    def read_config(self):
        self.difficulty_level = self.config["reward_prediction"]["difficulty_level"]
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

        self.apply_rnn = self.config["general"]["model"]["reward_predictor_apply_rnn"]

        self.apply_real_goal = self.config["reward_prediction"]["apply_real_goal"]

        # if self.difficulty_level == "general":
        #     self.data_paths = [self.config["graph_auto"]["data_path"] + '/general']
        #     self.max_game_count = float('inf')

        if self.difficulty_level == 'mixed':
            self.data_paths_dict = {}
            # self.data_paths_dict = [self.config["reward_prediction"]["data_path"] +
            #                         '/difficulty_level_{0}'.format(difficulty_level) for difficulty_level in
            #                         range(3, 10, 2)]
            self.dynamic_file_maps = {}
            counter = 0
            for difficulty_level in [3, 5, 7, 9]:
                self.data_paths_dict.update({difficulty_level: self.config["reward_prediction"]["data_path"] +
                                                               '/difficulty_level_{0}'.format(difficulty_level)})
                self.dynamic_file_maps.update({difficulty_level: {
                    "train": "train_all_dynamic_goal_{0}.json".format(self.agent.data_label[counter]),
                    "valid": "valid_dynamic_goal_{0}.json".format(self.agent.data_label[counter]),
                    "test": "test_dynamic_goal_{0}.json".format(self.agent.data_label[counter])
                }})
                counter += 1

            self.max_game_count = {'train': 25, 'valid': 5, 'test': 5}
        else:
            self.data_paths_dict = {self.difficulty_level: self.config["reward_prediction"]["data_path"] + \
                                                           "difficulty_level_{0}/".format(self.difficulty_level)}
            self.max_game_count = {'train': float('inf'), 'valid': float('inf'), 'test': float('inf')}
            self.dynamic_file_maps = {self.difficulty_level: {
                "train": "train_all_dynamic_goal_{0}.json".format(self.agent.data_label[0]),
                "valid": "valid_dynamic_goal_{0}.json".format(self.agent.data_label[0]),
                "test": "test_dynamic_goal_{0}.json".format(self.agent.data_label[0])
            }}

    def split_reset(self, split):
        if split == "train":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.train_sizes[data_type]})
            self.batch_size = self.training_batch_size
        elif split == "valid":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.valid_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size
        elif split == "test":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.test_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            # self.data = {"current_triplets": self.dataset[split]["current_triplets"][: self.use_this_many_data],
            #              "current_rewards": self.dataset[split]["current_rewards"][: self.use_this_many_data], }
            # self.data_size = self.use_this_many_data
            raise ValueError("unsupport fof the feature: use_this_many_data")
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0
        self.type_idx_pointer = 1

    def get_batch(self):
        indices = {}
        if self.split == "train":
            for data_type in self.all_data_types:
                indices.update({data_type: self.rng.choice(self.data_sizes[data_type],
                                                           int(self.training_batch_size / len(self.all_data_types)))})
        else:
            start = self.batch_pointer
            end = min(start + self.evaluate_batch_size, self.data_sizes[self.all_data_types[self.type_idx_pointer]])
            for data_type in self.all_data_types:
                if data_type == self.all_data_types[self.type_idx_pointer]:
                    indices.update({data_type: np.arange(start, end)})
                else:
                    indices.update({data_type: []})
            self.batch_pointer += self.evaluate_batch_size

            if self.batch_pointer >= self.data_sizes[self.all_data_types[self.type_idx_pointer]]:
                self.batch_pointer = 0
                self.type_idx_pointer -= 1
                # if self.type_idx_pointer < len(self.all_data_types) - 1:
                #     self.type_idx_pointer -= 1
                # else:
                #     self.type_idx_pointer = -1

        current_triplets, previous_triplets, \
        current_observations, previous_actions, current_rewards, \
        current_goal_sentences = \
            [], [], [], [], [], []
        decompress = self.dataset[self.split][self.all_data_types[0]]["graph_dataset"].decompress
        for data_type in self.all_data_types:
            for idx in indices[data_type]:
                current_triplets.append(decompress(self.data[data_type]["pred_current_triplets"][idx]))
                previous_triplets.append(decompress(self.data[data_type]["pred_previous_triplets"][idx]))
                current_rewards.append(self.data[data_type]["current_rewards"][idx])
                current_observations.append(self.data[data_type]["current_observation"][idx])
                previous_actions.append(self.data[data_type]["previous_action"][idx])

                if self.apply_real_goal:
                    current_goal_sentences.append(self.data[data_type]["current_goal_sentences_real"][idx])
                else:
                    current_goal_sentences.append(self.data[data_type]["current_goal_sentences"][idx])

                # current_goal_tuples.append(self.data[data_type]["current_goal_tuples"][idx])

        return current_triplets, previous_triplets, current_observations, previous_actions, \
               current_rewards, current_goal_sentences

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)


class RewardPredictionDynamicDataRNN(gym.Env):
    # FILENAMES_MAP = {
    #     "train": "train_all_goal.json",
    #     "valid": "valid_goal.json",
    #     "test": "test_goal.json"
    # }

    def __init__(self, config, agent, max_data_size=None, log_file=None):
        self.rng = None
        self.agent = agent
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.all_data_types = [0, 1]
        # Load dataset splits.
        self.data_sizes = {}
        self.train_sizes = {}
        self.valid_sizes = {}
        self.test_sizes = {}
        self.dataset = {}
        max_seq_length_dict = {}
        max_diff_tri_num_dict = {}
        avg_diff_tri_num_dict = {}
        # self.max_data_size = max_data_size

        self.dynamic_file_map = {
            "train": "train_all_dynamic_goal_{0}.json".format(self.agent.data_label),
            "valid": "valid_dynamic_goal_{0}.json".format(self.agent.data_label),
            "test": "test_dynamic_goal_{0}.json".format(self.agent.data_label)
        }

        for split in ["train", "valid", "test"]:  # "train", "valid", "test"
            self.dataset[split] = {}
            for data_type in self.all_data_types:
                self.dataset[split].update({data_type: {"truth_previous_triplets": [],
                                                        "pred_previous_triplets": [],
                                                        "truth_current_triplets": [],
                                                        "pred_current_triplets": [],
                                                        "current_rewards": [],
                                                        "previous_action": [],
                                                        "current_observation": [],
                                                        "current_goal_sentences": [],
                                                        "current_goal_sentences_real": []}})


            for difficulty_level in self.data_paths_dict.keys():
                max_seq_len, max_diff_tri_num, avg_diff_tri_num = self.load_dataset_for_reward_predict(
                    data_path=self.data_paths_dict[difficulty_level],
                    dynamic_file_name=self.dynamic_file_maps[difficulty_level][split],
                    split=split,
                    log_file=log_file)
                if split not in max_diff_tri_num_dict.keys():
                    max_diff_tri_num_dict.update({split: max_diff_tri_num})
                    avg_diff_tri_num_dict.update({split: avg_diff_tri_num})
                    max_seq_length_dict.update({split: max_seq_len})
                else:
                    if max_diff_tri_num > max_diff_tri_num_dict[split]:
                        max_diff_tri_num_dict.update({split: max_diff_tri_num})
                        avg_diff_tri_num_dict.update({split: avg_diff_tri_num})
                    if max_seq_len > max_seq_length_dict[split]:
                        max_seq_length_dict.update({split: max_seq_len})

        print("loaded dataset from {0} with max seq length {1} ...".format(self.data_paths_dict, max_seq_length_dict),
              file=log_file, flush=True)
        # print(
        #     "loaded dataset from {0} with max triplet difference {1} ...".format(self.data_path, max_diff_tri_num_dict),
        #     file=log_file, flush=True)
        # print(
        #     "loaded dataset from {0} with avg triplet difference {1} ...".format(self.data_path, avg_diff_tri_num_dict),
        #     file=log_file, flush=True)

        for data_type in self.all_data_types:
            self.train_sizes.update({data_type: len(self.dataset["train"][data_type]["current_rewards"])})
            self.valid_sizes.update({data_type: len(self.dataset["valid"][data_type]["current_rewards"])})
            self.test_sizes.update({data_type: len(self.dataset["test"][data_type]["current_rewards"])})
        # self.test_size = len(self.dataset["test"]["current_triplets"])
        self.batch_pointer = None
        self.type_idx_pointer = None
        self.batch_size, self.data = None, None
        self.split = "train"

    # def predict_triplets_from_dynamic(self, action, observation_string, input_adj_m, filter_mask, hx=None, cx=None):
    #     actions = [action]
    #     observations = [observation_string]
    #     with torch.no_grad():
    #         predicted_encodings, hx_new, cx_new, attn_mask, \
    #         input_node_name, input_relation_name, node_encodings, relation_encodings = \
    #             self.agent.compute_updated_dynamics(input_adj_m, actions, observations, hx, cx)
    #         pred_adj = self.agent.model.decode_graph(predicted_encodings, relation_encodings)
    #     filter_mask = np.repeat(filter_mask, len(input_adj_m), axis=0)
    #     adj_matrix = (to_np(pred_adj) > 0.5).astype(int)
    #     adj_matrix = filter_mask * adj_matrix
    #     triplets_pred = adj_to_triplets(adj_matrix=adj_matrix,
    #                                     node_vocab=self.agent.node_vocab,
    #                                     relation_vocab=self.agent.relation_vocab)
    #     triplets_input = adj_to_triplets(adj_matrix=input_adj_m,
    #                                      node_vocab=self.agent.node_vocab,
    #                                      relation_vocab=self.agent.relation_vocab)
    #     triplets_output = matching_object_from_obs(observations=observations,
    #                                                actions=actions,
    #                                                node_vocab=self.agent.node_vocab,
    #                                                pred_triplets=triplets_pred,
    #                                                input_triplets=triplets_input)
    #     pred_adj_matrix = self.agent.get_graph_adjacency_matrix(triplets_output)
    #
    #     return pred_adj_matrix, triplets_output[0]

    # def _dynamic_prediction(self, previous_triplets, example, obs, graph_dataset):
    #     tm1_tri_pred = previous_triplets
    #     previous_adj_m = self.agent.get_graph_adjacency_matrix([tm1_tri_pred])
    #     pred_adj_matrix, t_tri_pred = \
    #         self.predict_triplets_from_dynamic(action=example["previous_action"],
    #                                            observation_string=obs,
    #                                            input_adj_m=previous_adj_m,
    #                                            filter_mask=self.filter_mask,
    #                                            )
    #     tm1_tri_truth = graph_dataset.decompress(example["previous_graph_seen"])
    #     t_tri_truth = compute_current_triplets(tm1_tri_truth, example["target_commands"])
    #     diff_t_adj = diff_triplets(triplets1=t_tri_pred,
    #                                triplets2=t_tri_truth)
    #
    #     return tm1_tri_truth, tm1_tri_pred, t_tri_truth, t_tri_pred, diff_t_adj

    def load_dataset_for_reward_predict(self, data_path, dynamic_file_name, split, log_file):

        max_seq_len = 0
        max_diff_tri_num = 0
        total_pred_diff_num = 0
        total_samples = 0
        # file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        dynamic_data_path = pjoin(data_path, dynamic_file_name)
        desc = "Loading {}".format(os.path.basename(dynamic_data_path))
        print(desc, file=log_file, flush=True)
        with open(dynamic_data_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        # for data_type in self.all_data_types:
        self.dataset[split][self.all_data_types[0]]["graph_dataset"] = graph_dataset

        wt_tm1_tri_truth, wt_tm1_tri_pred, wt_t_tri_truth, wt_t_tri_pred, wt_r, wt_pre_a, wt_obs, \
        wt_goal_sen, wt_goal_sen_real, wt_step = None, None, None, None, None, None, None, None, None, None  # walkthrough
        ep_tm1_tri_truth, ep_tm1_tri_pred, ep_t_tri_truth, ep_t_tri_pred, ep_r, ep_pre_a, ep_obs, \
        ep_goal_sen, ep_goal_sen_real, ep_step = None, None, None, None, None, None, None, None, None, None  # exploration
        current_game_name = None
        pre_step_k = 1
        pre_step = None
        emphasize_num = 1
        pre_examined_goal = None
        game_count = 0
        for example in tqdm(data["examples"], desc=desc, file=log_file):
            # print(example["step"])
            # if self.max_data_size is not None:
            #     if total_samples > self.max_data_size:
            #         break
            if game_count > self.max_game_count[split]:
                break
            if " your score has just gone up by one point ." in example["observation"]:
                obs_bf_len = len(example["observation"])
                # print("find you.")
                # print(example["observation"])
                obs = example["observation"].replace(" your score has just gone up by one point .", "")
                obs_af_len = len(obs)
                assert obs_af_len < obs_bf_len
            else:
                obs = example["observation"]

            if current_game_name != example["game"]:
                wt_tm1_tri_truth, wt_tm1_tri_pred, wt_t_tri_truth, wt_t_tri_pred, wt_r, wt_pre_a, wt_obs, \
                wt_goal_sen, wt_goal_sen_real, wt_step = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}  # New game.
                current_game_name = example["game"]
                pre_examined_goal = None
                pre_step = [-1, -1, -1, -1]
                game_count += 1

            if example["step"][1] == 0 and example["step"][
                2] == 0:  # (i, k, j, goal_id) (wt step, branch_depth, branch_width)
                if pre_step[0] == example["step"][0] and example["step"][
                    3] == 0:  # exploration data for different goals
                    # if example["step"][3] == 0 and example["step"][0] != 0 or pre_step == example["step"]:
                    wk_step_back = True
                    ep_tm1_tri_truth, ep_tm1_tri_pred, ep_t_tri_truth, ep_t_tri_pred, ep_r, ep_pre_a, ep_obs, \
                    ep_goal_sen, ep_goal_sen_real, ep_step = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
                    # diff_t_adj = diff_triplets(
                    #     triplets1=graph_dataset.decompress(example["pred_current_triplets"]),
                    #     triplets2=graph_dataset.decompress(example["truth_current_triplets"]))
                    # total_pred_diff_num += len(diff_t_adj[0])
                    total_samples += 1

                    goal_sentence = pre_examined_goal
                    ep_tm1_tri_truth.update({goal_sentence: [example["truth_previous_triplets"]]})
                    ep_t_tri_truth.update({goal_sentence: [example["truth_current_triplets"]]})
                    ep_tm1_tri_pred.update({goal_sentence: [example["pred_previous_triplets"]]})
                    ep_t_tri_pred.update({goal_sentence: [example["pred_current_triplets"]]})
                    ep_r.update({goal_sentence: [example["current_reward"]]})
                    ep_pre_a.update({goal_sentence: [example["previous_action"]]})
                    ep_obs.update({goal_sentence: [obs]})
                    ep_goal_sen.update({goal_sentence: [example["current_goal_sentence"]]})
                    ep_goal_sen_real.update({goal_sentence: [example["current_goal_sentence_real"]]})
                    ep_step.update({goal_sentence: [example["step"]]})

                    # ep_tm1_tri_truth.append(example["truth_previous_triplets"])
                    # ep_t_tri_truth.append(example["truth_current_triplets"])
                    # ep_tm1_tri_pred.append(example["pred_previous_triplets"])
                    # ep_t_tri_pred.append(example["pred_current_triplets"])
                    #
                    # ep_r.append(example["current_reward"])
                    # ep_pre_a.append(example["previous_action"])
                    # ep_obs.append(obs)
                    # ep_goal_sen.append(example["current_goal_sentence"])

                else:  # adding walk through data
                    goal_sentence = example["current_goal_sentence_real"]
                    wk_step_back = False
                    # diff_t_adj = diff_triplets(
                    #     triplets1=graph_dataset.decompress(example["pred_current_triplets"]),
                    #     triplets2=graph_dataset.decompress(example["truth_current_triplets"]))
                    # total_pred_diff_num += len(diff_t_adj[0])

                    if example["current_reward"]:
                        pre_examined_goal = goal_sentence

                    if goal_sentence in wt_tm1_tri_truth.keys():
                        wt_tm1_tri_truth[goal_sentence] += [example["truth_previous_triplets"]]
                        wt_t_tri_truth[goal_sentence] += [example["truth_current_triplets"]]
                        wt_tm1_tri_pred[goal_sentence] += [example["pred_previous_triplets"]]
                        wt_t_tri_pred[goal_sentence] += [example["pred_current_triplets"]]
                        wt_r[goal_sentence] += [example["current_reward"]]
                        wt_pre_a[goal_sentence] += [example["previous_action"]]
                        wt_obs[goal_sentence] += [obs]
                        wt_goal_sen[goal_sentence] += [example["current_goal_sentence"]]
                        wt_goal_sen_real[goal_sentence] += [example["current_goal_sentence_real"]]
                        wt_step[goal_sentence] += [example["step"]]
                    else:
                        wt_tm1_tri_truth.update({goal_sentence: [example["truth_previous_triplets"]]})
                        wt_t_tri_truth.update({goal_sentence: [example["truth_current_triplets"]]})
                        wt_tm1_tri_pred.update({goal_sentence: [example["pred_previous_triplets"]]})
                        wt_t_tri_pred.update({goal_sentence: [example["pred_current_triplets"]]})
                        wt_r.update({goal_sentence: [example["current_reward"]]})
                        wt_pre_a.update({goal_sentence: [example["previous_action"]]})
                        wt_obs.update({goal_sentence: [obs]})
                        wt_goal_sen.update({goal_sentence: [example["current_goal_sentence"]]})
                        wt_goal_sen_real.update({goal_sentence: [example["current_goal_sentence_real"]]})
                        wt_step.update({goal_sentence: [example["step"]]})

                    ep_tm1_tri_truth, ep_tm1_tri_pred, ep_t_tri_truth, ep_t_tri_pred, ep_r, ep_pre_a, ep_obs, \
                    ep_goal_sen, ep_goal_sen_real, ep_step = \
                        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
                    if goal_sentence not in ep_tm1_tri_truth.keys():
                        ep_tm1_tri_truth.update({goal_sentence: []})
                        ep_t_tri_truth.update({goal_sentence: []})
                        ep_tm1_tri_pred.update({goal_sentence: []})
                        ep_t_tri_pred.update({goal_sentence: []})
                        ep_r.update({goal_sentence: []})
                        ep_pre_a.update({goal_sentence: []})
                        ep_obs.update({goal_sentence: []})
                        ep_goal_sen.update({goal_sentence: []})
                        ep_goal_sen_real.update({goal_sentence: []})
                        ep_step.update({goal_sentence: []})

                    # if example["current_goal_sentence"] == "eat the meal":
                    #     emphasize_num = 10  #
                    total_samples += 1
            else:
                wk_step_back = False
                if example["step"][1] <= pre_step_k and pre_step[2] != example["step"][2]:  # start to next trajectory
                    ep_tm1_tri_truth, ep_tm1_tri_pred, ep_t_tri_truth, ep_t_tri_pred, ep_r, ep_pre_a, ep_obs, \
                    ep_goal_sen, ep_goal_sen_real, ep_step = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
                    # previous_triplets = wt_t_tri_pred[-1]  # explore from the most recent walkthrough
                else:  # we are on the same trajectory, nothing need to be done
                    pass
                    # previous_triplets = ep_t_tri_pred[-1]  # explore from previous explore
                if example["step"][:3] != pre_step[:3]:  # going to the next step
                    pre_step_k = example["step"][1]

                goal_sentence = example["current_goal_sentence_real"]
                if goal_sentence not in wt_tm1_tri_truth.keys():
                    wt_tm1_tri_truth.update({goal_sentence: []})
                    wt_t_tri_truth.update({goal_sentence: []})
                    wt_tm1_tri_pred.update({goal_sentence: []})
                    wt_t_tri_pred.update({goal_sentence: []})
                    wt_r.update({goal_sentence: []})
                    wt_pre_a.update({goal_sentence: []})
                    wt_obs.update({goal_sentence: []})
                    wt_goal_sen.update({goal_sentence: []})
                    wt_goal_sen_real.update({goal_sentence: []})
                    wt_step.update({goal_sentence: []})

                if goal_sentence in ep_tm1_tri_truth.keys():
                    ep_tm1_tri_truth[goal_sentence] += [example["truth_previous_triplets"]]
                    ep_t_tri_truth[goal_sentence] += [example["truth_current_triplets"]]
                    ep_tm1_tri_pred[goal_sentence] += [example["pred_previous_triplets"]]
                    ep_t_tri_pred[goal_sentence] += [example["pred_current_triplets"]]
                    ep_r[goal_sentence] += [example["current_reward"]]
                    ep_pre_a[goal_sentence] += [example["previous_action"]]
                    ep_obs[goal_sentence] += [obs]
                    ep_goal_sen[goal_sentence] += [example["current_goal_sentence"]]
                    ep_goal_sen_real[goal_sentence] += [example["current_goal_sentence_real"]]
                    ep_step[goal_sentence] += [example["step"]]
                else:
                    ep_tm1_tri_truth.update({goal_sentence: [example["truth_previous_triplets"]]})
                    ep_t_tri_truth.update({goal_sentence: [example["truth_current_triplets"]]})
                    ep_tm1_tri_pred.update({goal_sentence: [example["pred_previous_triplets"]]})
                    ep_t_tri_pred.update({goal_sentence: [example["pred_current_triplets"]]})
                    ep_r.update({goal_sentence: [example["current_reward"]]})
                    ep_pre_a.update({goal_sentence: [example["previous_action"]]})
                    ep_obs.update({goal_sentence: [obs]})
                    ep_goal_sen.update({goal_sentence: [example["current_goal_sentence"]]})
                    ep_goal_sen_real.update({goal_sentence: [example["current_goal_sentence_real"]]})
                    ep_step.update({goal_sentence: [example["step"]]})

                # diff_t_adj = diff_triplets(triplets1=graph_dataset.decompress(example["pred_current_triplets"]),
                #                            triplets2=graph_dataset.decompress(example["truth_current_triplets"]))
                # total_pred_diff_num += len(diff_t_adj[0])
                # ep_tm1_tri_truth.append(example["truth_previous_triplets"])
                # ep_t_tri_truth.append(example["truth_current_triplets"])
                # ep_tm1_tri_pred.append(example["pred_previous_triplets"])
                # ep_t_tri_pred.append(example["pred_current_triplets"])
                #
                # ep_r.append(example["current_reward"])
                # ep_pre_a.append(example["previous_action"])
                # ep_obs.append(obs)
                # ep_goal_sen.append(example["current_goal_sentence"])
                # ep_goal_tuple.append(example["current_goal_tuples"])

                total_samples += 1

            data_type = example["current_reward"]
            # print(len(diff_t_adj))
            # if len(diff_t_adj[0]) > max_diff_tri_num:
            #     max_diff_tri_num = len(diff_t_adj[0])

            for i in range(emphasize_num):
                if wk_step_back:
                    # try:
                    #     self.dataset[split][data_type]["truth_previous_triplets"].append(
                    #         wt_tm1_tri_truth[goal_sentence][:-1] + ep_tm1_tri_truth[goal_sentence])
                    # except:
                    #     print(wt_tm1_tri_truth)
                    #     print(ep_tm1_tri_truth)
                    #     print(wt_r)
                    #     print(example)
                    #     raise ValueError('Debug')
                    self.dataset[split][data_type]["truth_previous_triplets"].append(
                        wt_tm1_tri_truth[goal_sentence][:-1] + ep_tm1_tri_truth[goal_sentence])
                    self.dataset[split][data_type]["pred_previous_triplets"].append(
                        wt_tm1_tri_pred[goal_sentence][:-1] + ep_tm1_tri_pred[goal_sentence])
                    self.dataset[split][data_type]["truth_current_triplets"].append(
                        wt_t_tri_truth[goal_sentence][:-1] + ep_t_tri_truth[goal_sentence])
                    self.dataset[split][data_type]["pred_current_triplets"].append(
                        wt_t_tri_pred[goal_sentence][:-1] + ep_t_tri_pred[goal_sentence])
                    self.dataset[split][data_type]["current_rewards"].append(
                        wt_r[goal_sentence][:-1] + ep_r[goal_sentence])
                    self.dataset[split][data_type]["previous_action"].append(
                        wt_pre_a[goal_sentence][:-1] + ep_pre_a[goal_sentence])
                    self.dataset[split][data_type]["current_observation"].append(
                        wt_obs[goal_sentence][:-1] + ep_obs[goal_sentence])
                    self.dataset[split][data_type]["current_goal_sentences"].append(
                        wt_goal_sen[goal_sentence][:-1] + ep_goal_sen[goal_sentence])
                    self.dataset[split][data_type]["current_goal_sentences_real"].append(
                        wt_goal_sen_real[goal_sentence][:-1] + ep_goal_sen_real[goal_sentence])
                    # self.dataset[split][data_type]["current_goal_tuples"].append(wt_goal_tuple[:-1] + ep_goal_tuple)
                else:
                    self.dataset[split][data_type]["truth_previous_triplets"].append(
                        wt_tm1_tri_truth[goal_sentence] + ep_tm1_tri_truth[goal_sentence])
                    self.dataset[split][data_type]["pred_previous_triplets"].append(
                        wt_tm1_tri_pred[goal_sentence] + ep_tm1_tri_pred[goal_sentence])
                    self.dataset[split][data_type]["truth_current_triplets"].append(
                        wt_t_tri_truth[goal_sentence] + ep_t_tri_truth[goal_sentence])
                    self.dataset[split][data_type]["pred_current_triplets"].append(
                        wt_t_tri_pred[goal_sentence] + ep_t_tri_pred[goal_sentence])
                    self.dataset[split][data_type]["current_rewards"].append(
                        wt_r[goal_sentence] + ep_r[goal_sentence])
                    self.dataset[split][data_type]["previous_action"].append(
                        wt_pre_a[goal_sentence] + ep_pre_a[goal_sentence])
                    self.dataset[split][data_type]["current_observation"].append(
                        wt_obs[goal_sentence] + ep_obs[goal_sentence])
                    self.dataset[split][data_type]["current_goal_sentences"].append(
                        wt_goal_sen[goal_sentence] + ep_goal_sen[goal_sentence])
                    self.dataset[split][data_type]["current_goal_sentences_real"].append(
                        wt_goal_sen_real[goal_sentence] + ep_goal_sen_real[goal_sentence])
                    # self.dataset[split][data_type]["current_goal_tuples"].append(wt_goal_tuple + ep_goal_tuple)
            emphasize_num = 1
            pre_step = example["step"]

            if len(wt_tm1_tri_truth[goal_sentence] + ep_tm1_tri_truth[goal_sentence]) > max_seq_len:
                max_seq_len = len(wt_tm1_tri_truth[goal_sentence] + ep_tm1_tri_truth[goal_sentence])
        avg_diff_tri_num = float(total_pred_diff_num) / total_samples
        return max_seq_len, max_diff_tri_num, avg_diff_tri_num

    def read_config(self):
        self.difficulty_level = self.config["reward_prediction"]["difficulty_level"]
        # self.data_path = self.config["reward_prediction"]["data_path"] + \
        #                  "difficulty_level_{0}/".format(self.difficulty_level)
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

        # self.apply_rnn = self.config["general"]["model"]["reward_predictor_apply_rnn"]
        self.apply_real_goal = self.config["reward_prediction"]["apply_real_goal"]

        if self.difficulty_level == 'mixed':
            self.data_paths_dict = {}
            # self.data_paths_dict = [self.config["reward_prediction"]["data_path"] +
            #                         '/difficulty_level_{0}'.format(difficulty_level) for difficulty_level in
            #                         range(3, 10, 2)]
            self.dynamic_file_maps = {}
            counter = 0
            for difficulty_level in [3, 5, 7, 9]:
                self.data_paths_dict.update({difficulty_level: self.config["reward_prediction"]["data_path"] +
                                                               '/difficulty_level_{0}'.format(difficulty_level)})
                self.dynamic_file_maps.update({difficulty_level: {
                    "train": "train_all_dynamic_goal_{0}.json".format(self.agent.data_label[counter]),
                    "valid": "valid_dynamic_goal_{0}.json".format(self.agent.data_label[counter]),
                    "test": "test_dynamic_goal_{0}.json".format(self.agent.data_label[counter])
                }})
                counter += 1

            self.max_game_count = {'train': 25, 'valid': 5, 'test': 5}
        else:
            self.data_paths_dict = {self.difficulty_level: self.config["reward_prediction"]["data_path"] + \
                                                           "difficulty_level_{0}/".format(self.difficulty_level)}
            self.max_game_count = {'train': float('inf'), 'valid': float('inf'), 'test': float('inf')}
            self.dynamic_file_maps = {self.difficulty_level: {
                "train": "train_all_dynamic_goal_{0}.json".format(self.agent.data_label[0]),
                "valid": "valid_dynamic_goal_{0}.json".format(self.agent.data_label[0]),
                "test": "test_dynamic_goal_{0}.json".format(self.agent.data_label[0])
            }}

    def split_reset(self, split):
        if split == "train":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.train_sizes[data_type]})
            self.batch_size = self.training_batch_size
        elif split == "valid":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.valid_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size
        elif split == "test":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.test_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            # self.data = {"current_triplets": self.dataset[split]["current_triplets"][: self.use_this_many_data],
            #              "current_rewards": self.dataset[split]["current_rewards"][: self.use_this_many_data], }
            # self.data_size = self.use_this_many_data
            raise ValueError("pls don't use the feature: use_this_many_data")
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0
        self.type_idx_pointer = 1

    def get_batch(self):
        indices = {}
        if self.split == "train":
            for data_type in self.all_data_types:
                indices.update({data_type: self.rng.choice(self.data_sizes[data_type],
                                                           int(self.training_batch_size / len(self.all_data_types)))})
        else:
            start = self.batch_pointer
            end = min(start + self.evaluate_batch_size, self.data_sizes[self.all_data_types[self.type_idx_pointer]])
            for data_type in self.all_data_types:
                if data_type == self.all_data_types[self.type_idx_pointer]:
                    indices.update({data_type: np.arange(start, end)})
                else:
                    indices.update({data_type: []})
            self.batch_pointer += self.evaluate_batch_size

            if self.batch_pointer >= self.data_sizes[self.all_data_types[self.type_idx_pointer]]:
                self.batch_pointer = 0
                self.type_idx_pointer -= 1
                # if self.type_idx_pointer < len(self.all_data_types) - 1:
                #     self.type_idx_pointer -= 1
                # else:
                #     self.type_idx_pointer = -1

        current_triplets, previous_triplets, \
        current_observations, previous_actions, current_rewards, \
        current_goal_sentences = \
            [], [], [], [], [], []
        decompress = self.dataset[self.split][self.all_data_types[0]]["graph_dataset"].decompress
        for data_type in self.all_data_types:
            for idx in indices[data_type]:
                # current_triplets.append(self.data[data_type]["pred_current_triplets"][idx])
                # previous_triplets.append(self.data[data_type]["pred_previous_triplets"][idx])

                s_current_triplets = [decompress(triplets) for triplets in
                                      self.data[data_type]["pred_current_triplets"][idx]]
                current_triplets.append(s_current_triplets)
                s_previous_triplets = [decompress(triplets) for triplets in
                                       self.data[data_type]["pred_previous_triplets"][idx]]
                previous_triplets.append(s_previous_triplets)

                current_rewards.append(self.data[data_type]["current_rewards"][idx])
                current_observations.append(self.data[data_type]["current_observation"][idx])
                previous_actions.append(self.data[data_type]["previous_action"][idx])
                if self.apply_real_goal:
                    current_goal_sentences.append(self.data[data_type]["current_goal_sentences_real"][idx])
                else:
                    current_goal_sentences.append(self.data[data_type]["current_goal_sentences"][idx])
                # current_goal_tuples.append(self.data[data_type]["current_goal_tuples"][idx])

        return current_triplets, previous_triplets, current_observations, previous_actions, \
               current_rewards, current_goal_sentences

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)


class RewardPredictionDataRNN(gym.Env):
    FILENAMES_MAP = {
        "train": "train_all_goal.json",
        "valid": "valid_goal.json",
        "test": "test_goal.json"
    }

    def __init__(self, config, log_file=None):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.all_data_types = [0, 1]
        # Load dataset splits.
        self.data_sizes = {}
        self.train_sizes = {}
        self.valid_sizes = {}
        self.test_sizes = {}
        self.dataset = {}
        max_seq_length_dict = {}
        for split in ["train", "valid", "test"]:  # "train", "valid", "test"
            self.dataset[split] = {}
            for data_type in self.all_data_types:
                self.dataset[split].update({data_type: {"previous_triplets": [],
                                                        "target_commands": [],
                                                        "current_rewards": [],
                                                        "previous_action": [],
                                                        "current_observation": [],
                                                        "current_goal_sentences": [],
                                                        "current_goal_tuples": []}})
            max_seq_len = self.load_dataset_for_reward_predict(split, log_file)
            max_seq_length_dict.update({split: max_seq_len})
        print("loaded dataset from {0} with max seq length {1} ...".format(self.data_path, max_seq_length_dict),
              file=log_file, flush=True)
        for data_type in self.all_data_types:
            self.train_sizes.update({data_type: len(self.dataset["train"][data_type]["current_rewards"])})
            self.valid_sizes.update({data_type: len(self.dataset["valid"][data_type]["current_rewards"])})
            self.test_sizes.update({data_type: len(self.dataset["test"][data_type]["current_rewards"])})
        # self.test_size = len(self.dataset["test"]["current_triplets"])
        self.batch_pointer = None
        self.type_idx_pointer = None
        self.batch_size, self.data = None, None
        self.split = "train"

    def load_dataset_for_reward_predict(self, split, log_file):
        max_seq_len = 0
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        desc = "Loading {}".format(os.path.basename(file_path))
        print(desc, file=log_file, flush=True)
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        # for data_type in self.all_data_types:
        self.dataset[split][self.all_data_types[0]]["graph_dataset"] = graph_dataset

        wt_pre_tri, wt_tgt_cmm, wt_r, wt_pre_a, wt_obs, wt_goal_sen, wt_goal_tuple = \
            [], [], [], [], [], [], []  # New game.
        ep_pre_tri, ep_tgt_cmm, ep_r, ep_pre_a, ep_obs, ep_goal_sen, ep_goal_tuple = \
            None, None, None, None, None, None, None
        current_game_name = None
        pre_step_k = 1
        pre_step = None
        emphasize_num = 1
        for example in tqdm(data["examples"], desc=desc, file=log_file):
            # print(example["step"])

            if " your score has just gone up by one point ." in example["observation"]:
                obs_bf_len = len(example["observation"])
                # print("find you.")
                # print(example["observation"])
                obs = example["observation"].replace(" your score has just gone up by one point .", "")
                obs_af_len = len(obs)
                assert obs_af_len < obs_bf_len
            else:
                obs = example["observation"]

            if current_game_name != example["game"]:
                wt_pre_tri, wt_tgt_cmm, wt_r, wt_pre_a, wt_obs, wt_goal_sen, wt_goal_tuple = \
                    [], [], [], [], [], [], []  # New game.
                current_game_name = example["game"]
                pre_step = [-1, -1, -1, -1]

            if example["step"][1] == 0 and example["step"][
                2] == 0:  # (i, k, j) (walkthrough step, branching_depth, branching_width)
                if pre_step[0] == example["step"][0] and example["step"][
                    3] == 0:  # exploration data for different goals
                    wk_step_back = True
                    ep_pre_tri, ep_tgt_cmm, ep_r, ep_pre_a, ep_obs, ep_goal_sen, ep_goal_tuple = \
                        [], [], [], [], [], [], []
                    ep_pre_tri.append(example["previous_graph_seen"])
                    ep_tgt_cmm.append(example["target_commands"])
                    ep_r.append(example["current_reward"])
                    ep_pre_a.append(example["previous_action"])
                    ep_obs.append(obs)
                    ep_goal_sen.append(example["current_goal_sentence"])
                    # ep_goal_tuple.append(example["current_goal_tuples"])
                else:  # adding walk through data
                    wk_step_back = False
                    if example["step"][:3] != pre_step[:3]:  # going to the next step
                        wt_pre_tri.append(example["previous_graph_seen"])
                        wt_tgt_cmm.append(example["target_commands"])
                        wt_r.append(example["current_reward"])
                        wt_pre_a.append(example["previous_action"])
                        wt_obs.append(obs)
                        wt_goal_sen.append(example["current_goal_sentence"])
                    else:  # we are still on the same step, but have different goal
                        wt_goal_sen[-1] += "&" + example["current_goal_sentence"]
                    # wt_goal_tuple.append(example["current_goal_tuples"])
                    ep_pre_tri, ep_tgt_cmm, ep_r, ep_pre_a, ep_obs, ep_goal_sen, ep_goal_tuple = \
                        [], [], [], [], [], [], []
                    if example["current_goal_sentence"] == "eat the meal":
                        emphasize_num = 10  #
            else:
                wk_step_back = False
                if example["step"][1] <= pre_step_k and pre_step[2] != example["step"][2]:  # start to next trajectory
                    ep_pre_tri, ep_tgt_cmm, ep_r, ep_pre_a, ep_obs, ep_goal_sen, ep_goal_tuple = \
                        [], [], [], [], [], [], []
                if example["step"][:3] != pre_step[:3]:  # going to the next step
                    pre_step_k = example["step"][1]
                    ep_pre_tri.append(example["previous_graph_seen"])
                    ep_tgt_cmm.append(example["target_commands"])
                    ep_r.append(example["current_reward"])
                    ep_pre_a.append(example["previous_action"])
                    ep_obs.append(obs)
                    ep_goal_sen.append(example["current_goal_sentence"])
                    # ep_goal_tuple.append(example["current_goal_tuples"])
                else:  # we are still on the same step, but have different goal
                    ep_goal_sen[-1] += " & " + example["current_goal_sentence"]

            data_type = example["current_reward"]
            for i in range(emphasize_num):
                if wk_step_back:
                    self.dataset[split][data_type]["previous_triplets"].append(wt_pre_tri[:-1] + ep_pre_tri)
                    self.dataset[split][data_type]["target_commands"].append(wt_tgt_cmm[:-1] + ep_tgt_cmm)
                    self.dataset[split][data_type]["current_rewards"].append(wt_r[:-1] + ep_r)
                    self.dataset[split][data_type]["previous_action"].append(wt_pre_a[:-1] + ep_pre_a)
                    self.dataset[split][data_type]["current_observation"].append(wt_obs[:-1] + ep_obs)
                    self.dataset[split][data_type]["current_goal_sentences"].append(wt_goal_sen[:-1] + ep_goal_sen)
                else:
                    self.dataset[split][data_type]["previous_triplets"].append(wt_pre_tri + ep_pre_tri)
                    self.dataset[split][data_type]["target_commands"].append(wt_tgt_cmm + ep_tgt_cmm)
                    self.dataset[split][data_type]["current_rewards"].append(wt_r + ep_r)
                    self.dataset[split][data_type]["previous_action"].append(wt_pre_a + ep_pre_a)
                    self.dataset[split][data_type]["current_observation"].append(wt_obs + ep_obs)
                    self.dataset[split][data_type]["current_goal_sentences"].append(wt_goal_sen + ep_goal_sen)
                    # self.dataset[split][data_type]["current_goal_tuples"].append(wt_goal_tuple + ep_goal_tuple)
                emphasize_num = 1
            pre_step = example["step"]

            if len(wt_pre_tri + ep_pre_tri) > max_seq_len:
                max_seq_len = len(wt_pre_tri + ep_pre_tri)
        return max_seq_len

    def read_config(self):
        self.difficulty_level = self.config["reward_prediction"]["difficulty_level"]
        if self.difficulty_level == 7 or self.difficulty_level == 9:
            raise EnvironmentError("recheck the code here")
        self.data_path = self.config["reward_prediction"]["data_path"] + \
                         "difficulty_level_{0}/".format(self.difficulty_level)
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

        # self.apply_rnn = self.config["general"]["model"]["reward_predictor_apply_rnn"]

    def split_reset(self, split):
        if split == "train":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.train_sizes[data_type]})
            self.batch_size = self.training_batch_size
        elif split == "valid":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.valid_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size
        elif split == "test":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.test_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            # self.data = {"current_triplets": self.dataset[split]["current_triplets"][: self.use_this_many_data],
            #              "current_rewards": self.dataset[split]["current_rewards"][: self.use_this_many_data], }
            # self.data_size = self.use_this_many_data
            raise ValueError("unsupport fof the feature: use_this_many_data")
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0
        self.type_idx_pointer = 0

    def get_batch(self):
        indices = {}
        if self.split == "train":
            for data_type in self.all_data_types:
                indices.update({data_type: self.rng.choice(self.data_sizes[data_type],
                                                           int(self.training_batch_size / len(self.all_data_types)))})
        else:
            start = self.batch_pointer
            end = min(start + self.evaluate_batch_size, self.data_sizes[self.all_data_types[self.type_idx_pointer]])
            for data_type in self.all_data_types:
                if data_type == self.all_data_types[self.type_idx_pointer]:
                    indices.update({data_type: np.arange(start, end)})
                else:
                    indices.update({data_type: []})
            self.batch_pointer += self.evaluate_batch_size

            if self.batch_pointer >= self.data_sizes[self.all_data_types[self.type_idx_pointer]]:
                self.batch_pointer = 0
                if self.type_idx_pointer < len(self.all_data_types) - 1:
                    self.type_idx_pointer += 1
                else:
                    self.type_idx_pointer = -1

        current_triplets, previous_triplets, \
        current_observations, previous_actions, current_rewards, \
        current_goal_sentences = \
            [], [], [], [], [], []
        decompress = self.dataset[self.split][self.all_data_types[0]]["graph_dataset"].decompress
        for data_type in self.all_data_types:
            for idx in indices[data_type]:
                s_previous_triplets = [decompress(triplets) for triplets in
                                       self.data[data_type]["previous_triplets"][idx]]
                s_target_command = self.data[data_type]["target_commands"][idx]
                assert len(s_previous_triplets) == len(s_target_command)
                current_triplets.append(
                    [compute_current_triplets(s_previous_triplets[j], s_target_command[j]) for j in
                     range(len(s_previous_triplets))])
                previous_triplets.append(s_previous_triplets)
                current_rewards.append(self.data[data_type]["current_rewards"][idx])
                current_observations.append(self.data[data_type]["current_observation"][idx])
                previous_actions.append(self.data[data_type]["previous_action"][idx])
                current_goal_sentences.append(self.data[data_type]["current_goal_sentences"][idx])
                # current_goal_tuples.append(self.data[data_type]["current_goal_tuples"][idx])

        return current_triplets, previous_triplets, current_observations, previous_actions, \
               current_rewards, current_goal_sentences

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)


class RewardPredictionDataGoal(gym.Env):
    FILENAMES_MAP = {
        "train": "train_goal.json",
        "valid": "valid_goal.json",
        "test": "test_goal.json"
    }

    def __init__(self, config, log_file=None):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.all_data_types = [0, 1]
        # Load dataset splits.
        self.data_sizes = {}
        self.train_sizes = {}
        self.valid_sizes = {}
        self.test_sizes = {}
        self.dataset = {}
        for split in ["train", "valid"]:  # "train", "valid", "test"
            self.dataset[split] = {}
            for data_type in self.all_data_types:
                self.dataset[split].update({data_type: {"previous_triplets": [],
                                                        "target_commands": [],
                                                        "current_rewards": [],
                                                        "previous_action": [],
                                                        "current_observation": [],
                                                        "current_goal_sentences": [],
                                                        "current_goal_tuples": []}})
            self.load_dataset_for_reward_predict(split, log_file)
        print("loaded dataset from {} ...".format(self.data_path), file=log_file, flush=True)
        for data_type in self.all_data_types:
            self.train_sizes.update({data_type: len(self.dataset["train"][data_type]["current_rewards"])})
            self.valid_sizes.update({data_type: len(self.dataset["valid"][data_type]["current_rewards"])})
            self.valid_sizes.update({data_type: len(self.dataset["test"][data_type]["current_rewards"])})
        # self.test_size = len(self.dataset["test"]["current_triplets"])
        self.batch_pointer = None
        self.type_idx_pointer = None
        self.batch_size, self.data = None, None
        self.split = "train"

    def load_dataset_for_reward_predict(self, split, log_file):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        desc = "Loading {}".format(os.path.basename(file_path))
        print(desc, file=log_file, flush=True)
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        # for data_type in self.all_data_types:
        self.dataset[split][self.all_data_types[0]]["graph_dataset"] = graph_dataset

        current_game_name = None
        for example in tqdm(data["examples"], desc=desc, file=log_file):

            if example["current_reward"] == 1:
                obs_bf_len = len(example["observation"])
                # print("find you.")
                # print(example["observation"])
                if " your score has just gone up by one point ." in example["observation"]:
                    obs = example["observation"].replace(" your score has just gone up by one point .", "")
                    obs_af_len = len(obs)
                    assert obs_af_len < obs_bf_len
                else:
                    obs = example["observation"]
            else:
                obs = example["observation"]

            data_type = example["current_reward"]
            self.dataset[split][data_type]["previous_triplets"].append(example["previous_graph_seen"])
            self.dataset[split][data_type]["target_commands"].append(example["target_commands"])
            self.dataset[split][data_type]["current_rewards"].append(example["current_reward"])
            self.dataset[split][data_type]["previous_action"].append(example["previous_action"])
            self.dataset[split][data_type]["current_goal_sentences"].append(example["current_goal_sentence"])
            self.dataset[split][data_type]["current_goal_tuples"].append(example["current_goal_tuples"])
            self.dataset[split][data_type]["current_observation"].append(obs)

    def read_config(self):
        self.difficulty_level = self.config["reward_prediction"]["difficulty_level"]
        self.data_path = self.config["reward_prediction"]["data_path"] + \
                         "difficulty_level_{0}/".format(self.difficulty_level)
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

        self.apply_rnn = self.config["general"]["model"]["apply_rnn"]

    def split_reset(self, split):
        if split == "train":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.train_sizes[data_type]})
            self.batch_size = self.training_batch_size
        elif split == "valid":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.valid_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size
        # elif split == "test":
        #     for data_type in self.all_data_types:
        #         self.data_sizes.update({data_type: self.test_sizes[data_type]})
        #     self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            # self.data = {"current_triplets": self.dataset[split]["current_triplets"][: self.use_this_many_data],
            #              "current_rewards": self.dataset[split]["current_rewards"][: self.use_this_many_data], }
            # self.data_size = self.use_this_many_data
            raise ValueError("unsupport fof the feature: use_this_many_data")
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0
        self.type_idx_pointer = 0

    def get_batch(self):
        indices = {}
        if self.split == "train":
            for data_type in self.all_data_types:
                indices.update({data_type: self.rng.choice(self.data_sizes[data_type],
                                                           int(self.training_batch_size / len(self.all_data_types)))})
        else:
            start = self.batch_pointer
            end = min(start + self.evaluate_batch_size, self.data_sizes[self.all_data_types[self.type_idx_pointer]])
            for data_type in self.all_data_types:
                if data_type == self.all_data_types[self.type_idx_pointer]:
                    indices.update({data_type: np.arange(start, end)})
                else:
                    indices.update({data_type: []})
            self.batch_pointer += self.evaluate_batch_size

            if self.batch_pointer >= self.data_sizes[self.all_data_types[self.type_idx_pointer]]:
                self.batch_pointer = 0
                if self.type_idx_pointer < len(self.all_data_types) - 1:
                    self.type_idx_pointer += 1
                else:
                    self.type_idx_pointer = -1

        current_triplets, current_observations, previous_actions, current_rewards, \
        current_goal_sentences, current_goal_tuples = [], [], [], [], [], []
        decompress = self.dataset[self.split][self.all_data_types[0]]["graph_dataset"].decompress
        for data_type in self.all_data_types:
            for idx in indices[data_type]:
                s_previous_triplets = decompress(self.data[data_type]["previous_triplets"][idx])
                s_target_command = self.data[data_type]["target_commands"][idx]
                current_triplets.append(compute_current_triplets(s_previous_triplets, s_target_command))
                current_rewards.append(self.data[data_type]["current_rewards"][idx])
                current_observations.append(self.data[data_type]["current_observation"][idx])
                previous_actions.append(self.data[data_type]["previous_action"][idx])
                current_goal_sentences.append(self.data[data_type]["current_goal_sentences"][idx])
                current_goal_tuples.append(self.data[data_type]["current_goal_tuples"][idx])

        return current_triplets, current_observations, previous_actions, \
               current_rewards, current_goal_sentences, current_goal_tuples

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
