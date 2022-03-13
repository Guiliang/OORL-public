import json
import os
from os.path import join as pjoin
from tqdm import tqdm
import gym
import numpy as np
from generic.data_utils import GraphDataset


class SPData(gym.Env):
    FILENAMES_MAP = {
        "full": {
            "train": "train.full.json",
            "valid": "valid.full.json",
            "test": "test.full.json"
        },
        "seen": {
            "train": "train.seen.json",
            "valid": "valid.seen.json",
            "test": "test.seen.json"
        }
    }

    def __init__(self, config, seed=None, log_file=None):
        self.rng = None
        self.config = config
        self.read_config()
        if seed is None:
            self.seed(self.random_seed)
        else:
            self.seed(seed)
        self.log_file = log_file

        # Load dataset splits.
        self.dataset = {}
        for split in ["train", "valid"]:  # "train", "valid", "test"
            self.dataset[split] = {
                "current_graph": [],
                "previous_graph": [],
                "action": [],
                "observation": [],
                "target_commands": []
            }
            game_count_datasets = 0
            for data_path in self.data_paths:
                game_count = self.load_dataset_for_sp(data_path=data_path, split=split)
                game_count_datasets += game_count

            print('{0} dataset has {1} games'.format(split, game_count_datasets), file=log_file)

        self.train_size = len(self.dataset["train"]["current_graph"])
        self.valid_size = len(self.dataset["valid"]["current_graph"])
        # self.test_size = len(self.dataset["test"]["current_graph"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def load_dataset_for_sp(self, data_path, split):
        file_path = pjoin(data_path, self.FILENAMES_MAP[self.graph_type][split])
        print("loaded dataset from {} ...".format(file_path), file=self.log_file, flush=True)
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        self.dataset[split]["graph_dataset"] = graph_dataset

        game_count = 0
        prev_game_name = None
        desc = "Loading {}".format(os.path.basename(file_path))
        for example in tqdm(data["examples"], desc=desc, file=self.log_file):
            if example['game'] != prev_game_name:
                game_count += 1
                prev_game_name = example['game']
            if game_count > self.max_game_count[split]:
                break
            if " your score has just gone up by one point ." in example["observation"]:
                obs_bf_len = len(example["observation"])
                obs = example["observation"].replace(" your score has just gone up by one point .", "")
                obs_af_len = len(obs)
                assert obs_af_len < obs_bf_len
            else:
                obs = example["observation"]

            self.dataset[split]["current_graph"].append(example["current_graph"])
            self.dataset[split]["previous_graph"].append(example["previous_graph"])
            self.dataset[split]["action"].append(example["action"])
            self.dataset[split]["observation"].append(obs)
            self.dataset[split]["target_commands"].append(example["target_commands"])

        return game_count

    def read_config(self):
        self.graph_type = self.config["graph_auto"]["graph_type"]
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

        self.difficulty_level = self.config["graph_auto"]["difficulty_level"]
        if self.difficulty_level == "general":
            self.data_paths = [self.config["graph_auto"]["data_path"] + '/general']
            self.max_game_count = {'train': float('inf'), 'valid': float('inf'), 'test': float('inf')}
        elif self.difficulty_level == 'mixed':  # the training data comes from multiple files
            self.data_paths = [self.config["graph_auto"]["data_path"] +
                               '/difficulty_level_{0}'.format(difficulty_level) for difficulty_level in range(3, 10, 2)]
            self.max_game_count = {'train': 25, 'valid': 5, 'test': 5}
        else:
            self.data_paths = [self.config["graph_auto"]["data_path"] +
                               '/difficulty_level_{0}'.format(self.difficulty_level)]
            self.max_game_count = {'train': float('inf'), 'valid': float('inf'), 'test': float('inf')}

    def split_reset(self, split):
        if split == "train":
            self.data_size = self.train_size
            self.batch_size = self.training_batch_size
        elif split == "valid":
            self.data_size = self.valid_size
            self.batch_size = self.evaluate_batch_size
        else:
            self.data_size = self.test_size
            self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            self.data = {"current_graph": self.dataset[split]["current_graph"][: self.use_this_many_data],
                         "previous_graph": self.dataset[split]["previous_graph"][: self.use_this_many_data],
                         "action": self.dataset[split]["action"][: self.use_this_many_data],
                         "observation": self.dataset[split]["observation"][: self.use_this_many_data],
                         "target_commands": self.dataset[split]["target_commands"][: self.use_this_many_data],
                         }
            self.data_size = self.use_this_many_data
        else:
            self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0

    def get_batch(self):
        if self.split == "train":
            indices = self.rng.choice(self.data_size, self.training_batch_size)
        else:
            start = self.batch_pointer
            end = min(start + self.evaluate_batch_size, self.data_size)
            indices = np.arange(start, end)
            self.batch_pointer += self.evaluate_batch_size

            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0

        target_graph, previous_graph, action, observation = [], [], [], []
        decompress = self.dataset[self.split]["graph_dataset"].decompress
        for idx in indices:
            action.append(self.data["action"][idx])
            # Perform just-in-time decompression.
            target_graph.append(decompress(self.data["current_graph"][idx]))
            previous_graph.append(decompress(self.data["previous_graph"][idx]))
            observation.append(self.data["observation"][idx])

        return target_graph, previous_graph, action, observation

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)


class SPDataUnSupervised(gym.Env):
    FILENAMES_MAP = {
        "full": {
            "train": "train_unsupervised.full.json",
            "valid": "valid_unsupervised.full.json",
            "test": "test_unsupervised.full.json"
        },
        "seen": {
            "train": "train_unsupervised.seen.json",
            "valid": "valid_unsupervised.seen.json",
            "test": "test_unsupervised.seen.json"
        }
    }

    def __init__(self, config, log_file=None):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.log_file = log_file
        self.all_data_types = [0, 1]
        self.data_sizes = {}
        self.train_sizes = {}
        self.valid_sizes = {}
        self.test_sizes = {}

        # Load dataset splits.
        self.dataset = {}
        max_seq_len_dict = {}
        for split in ["train", "valid"]:  # "train", "valid", "test"]:

            self.dataset[split] = {}
            for data_type in self.all_data_types:
                self.dataset[split].update({data_type: {
                    "target_commands": [],
                    "previous_graph": [],
                    "current_graph": [],
                    "action": [],
                    "current_observation": [],
                    "reward": []
                }})
            max_seq_len = self.load_dataset_for_sp(split)
            max_seq_len_dict.update({split: max_seq_len})

        print("loaded dataset from {0} with max seq length {1} ...".format(self.data_path, max_seq_len_dict),
              file=log_file, flush=True)

        for data_type in self.all_data_types:
            self.train_sizes.update({data_type: len(self.dataset["train"][data_type]["target_commands"])})
            self.valid_sizes.update({data_type: len(self.dataset["valid"][data_type]["target_commands"])})
            # self.test_sizes.update({data_type: len(self.dataset["test"][data_type]["target_commands"])})
        print(self.train_sizes)
        print(self.valid_sizes)
        self.batch_pointer = None
        self.batch_size, self.data = None, None
        self.split = "train"

    def load_dataset_for_sp(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[self.graph_type][split])
        print("loaded dataset from {} ...".format(file_path), file=self.log_file, flush=True)
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        self.dataset[split]["graph_dataset"] = graph_dataset
        max_seq_length_index = None
        desc = "Loading {}".format(os.path.basename(file_path))
        wt_tgt_cmd, wt_pre_graph, wt_cur_graph, wt_pre_a, wt_obs, wt_r = [], [], [], [], [], []  # walkthroughs
        ep_tgt_cmd, ep_pre_graph, ep_cur_graph, ep_pre_a, ep_obs, ep_r = None, None, None, None, None, None  # explorations
        current_game_name = None
        pre_step_k = 1
        max_seq_len = 0
        for example in tqdm(data["examples"], desc=desc, file=self.log_file):
            if " your score has just gone up by one point ." in example["observation"]:
                obs_bf_len = len(example["observation"])
                obs = example["observation"].replace(" your score has just gone up by one point .", "")
                obs_af_len = len(obs)
                assert obs_af_len < obs_bf_len
            else:
                obs = example["observation"]
            # print(example["step"])
            if current_game_name != example["game"]:
                wt_tgt_cmd, wt_pre_graph, wt_cur_graph, wt_pre_a, wt_obs, wt_r = [], [], [], [], [], []  # New game.
                current_game_name = example["game"]
            if example["step"][1] == 0 and example["step"][
                2] == 0:  # (i, k, j) (walkthrough step, branching_depth, branching_width)
                wt_tgt_cmd.append(example["target_commands"])
                wt_pre_graph.append(example["previous_graph"])
                wt_cur_graph.append(example["current_graph"])
                wt_pre_a.append(example["action"])
                wt_obs.append(obs)
                wt_r.append(example["reward"])
                ep_tgt_cmd, ep_pre_graph, ep_cur_graph, ep_pre_a, ep_obs, ep_r = \
                    [], [], [], [], [], []
            else:
                if example["step"][1] <= pre_step_k:
                    ep_tgt_cmd, ep_pre_graph, ep_cur_graph, ep_pre_a, ep_obs, ep_r = \
                        [], [], [], [], [], []  # explores
                pre_step_k = example["step"][1]
                ep_tgt_cmd.append(example["target_commands"])
                ep_pre_graph.append(example["previous_graph"])
                ep_cur_graph.append(example["current_graph"])
                ep_pre_a.append(example["action"])
                # ep_graph_choice.append(example["graph_choices"])
                ep_obs.append(obs)
                ep_r.append(example["reward"])

            data_type = example["reward"]
            self.dataset[split][data_type]["target_commands"].append(wt_tgt_cmd + ep_tgt_cmd)
            self.dataset[split][data_type]["previous_graph"].append(wt_pre_graph + ep_pre_graph)
            self.dataset[split][data_type]["current_graph"].append(wt_cur_graph + ep_cur_graph)
            self.dataset[split][data_type]["action"].append(wt_pre_a + ep_pre_a)
            self.dataset[split][data_type]["current_observation"].append(wt_obs + ep_obs)
            self.dataset[split][data_type]["reward"].append(wt_r + ep_r)
            if len(wt_tgt_cmd + ep_tgt_cmd) > max_seq_len:
                max_seq_len = len(wt_tgt_cmd + ep_tgt_cmd)
                # max_seq_length_index = len(self.dataset[split][data_type]["target_commands"]) - 1
        # print(self.dataset[split]["action"][max_seq_length_index])
        # print(self.dataset[split]["current_observation"][max_seq_length_index])
        return max_seq_len

    def read_config(self):
        self.graph_type = self.config["graph_auto"]["graph_type"]
        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

        self.difficulty_level = self.config["graph_auto"]["difficulty_level"]
        if self.difficulty_level == "general":
            self.data_path = self.config["graph_auto"]["data_path"] + '/general'
        else:
            self.data_path = self.config["graph_auto"]["data_path"] + '/difficulty_level_{0}'.format(
                self.difficulty_level)

    def split_reset(self, split):
        if split == "train":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.train_sizes[data_type]})
            self.batch_size = self.training_batch_size
        elif split == "valid":
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.valid_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size
        else:
            for data_type in self.all_data_types:
                self.data_sizes.update({data_type: self.test_sizes[data_type]})
            self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            # self.data = {"target_commands": self.dataset[split]["target_commands"][: self.use_this_many_data],
            #              "target_graph": self.dataset[split]["target_graph"][: self.use_this_many_data],
            #              "previous_graph": self.dataset[split]["previous_graph"][: self.use_this_many_data],
            #              "action": self.dataset[split]["action"][: self.use_this_many_data],
            #              "current_observation": self.dataset[split]["current_observation"][: self.use_this_many_data]}
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

        target_graph_triplets, previous_graph_triplets, action, current_observations, rewards = \
            [], [], [], [], []
        decompress = self.dataset[self.split]["graph_dataset"].decompress

        for data_type in self.all_data_types:
            for idx in indices[data_type]:
                s_previous_triplets = [decompress(triplets) for triplets in
                                       self.data[data_type]["previous_graph"][idx]]
                previous_graph_triplets.append(s_previous_triplets)
                s_target_command = self.data[data_type]["target_commands"][idx]
                s_target_graph_triplets = [decompress(triplets) for triplets in
                                           self.data[data_type]["current_graph"][idx]]
                target_graph_triplets.append(s_target_graph_triplets)
                action.append(self.data[data_type]["action"][idx])
                current_observations.append(self.data[data_type]["current_observation"][idx])
                rewards.append(self.data[data_type]["reward"][idx])

        return target_graph_triplets, previous_graph_triplets, action, current_observations, rewards

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
