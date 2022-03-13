# part of the code are from https://github.com/hill-a/stable-baselines/
import random
from collections import namedtuple
import numpy as np
import torch
from generic.model_utils import to_np
from generic.segment_tree import SumSegmentTree, MinSegmentTree

# a snapshot of state to be stored in replay memory

ReplayRecordSuperRL = namedtuple('Transition', ('step_label', 'observation_item', 'goal_item', 'prev_action_item',
                                                'action_candidate_item', 'chosen_indices', 'graph_triplets',
                                                'reward', 'graph_reward', 'count_reward', 'is_final'))
ReplayRecordRL = namedtuple('Transition', ('step_label', 'observation_item', 'goal_item', 'prev_action_item',
                                                  'action_candidate_item', 'chosen_indices',
                                                  'predicted_encoding', 'predicted_mask',
                                                  'reward', 'graph_reward', 'count_reward', 'is_final'))

TransitionRecordSuperRL = namedtuple('Transition', ('observation_item', 'graph_triplets', 'goal_item',
                                                    'selected_action_item', 'next_reward',
                                                    'next_observation_item', 'next_graph_triplets',))

TransitionRecordUnSuperRL = namedtuple('Transition', ('observation_item',
                                                      'predicted_encoding', 'predicted_mask', 'goal_item',
                                                      'selected_action_item', 'next_reward',
                                                      'next_observation_item',))


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, discount_gamma_game_reward=1.0,
                 discount_gamma_graph_reward=1.0, discount_gamma_count_reward=1.0,
                 accumulate_reward_from_final=False, supervised_rl_flag=True):
        # prioritized replay memory
        self._storage = []
        self.capacity = capacity
        self._next_idx = 0

        assert priority_fraction >= 0
        self._alpha = priority_fraction

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.discount_gamma_game_reward = discount_gamma_game_reward
        self.discount_gamma_graph_reward = discount_gamma_graph_reward
        self.discount_gamma_count_reward = discount_gamma_count_reward
        self.accumulate_reward_from_final = accumulate_reward_from_final
        self.supervised_rl_flag = supervised_rl_flag

        self.data_store_step_label = 0  # applied to judge which is the next step in memory

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self.capacity

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, *args):
        """
        add a new transition to the buffer
        """
        idx = self._next_idx
        # if self.supervised_rl_flag:
        #     data = ReplayRecordSuperRL(*args)
        # else:
        data = ReplayRecordRL(*args)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self.capacity
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def get_next_final_pos(self, which_memory, head):
        i = head
        while True:
            if i >= len(self._storage):
                return None
            if self._storage[i].is_final:
                return i
            i += 1
        return None

    def _get_single_transition(self, idx, n):
        assert n > 0
        head = idx
        # if n is 1, then head can't be is_final
        if n == 1:
            if self._storage[head].is_final:
                return None
        #  if n > 1, then all except tail can't be is_final
        else:
            if np.any([item.is_final for item in self._storage[head: head + n]]):
                return None

        next_final = self.get_next_final_pos(self._storage, head)
        if next_final is None:
            return None

        # all good
        obs = self._storage[head].observation_item
        prev_action = self._storage[head].prev_action_item
        goal_sentence = self._storage[head].goal_item
        candidate = self._storage[head].action_candidate_item
        chosen_indices = self._storage[head].chosen_indices
        # if self.supervised_rl_flag:
        #     graph_triplets = self._storage[head].graph_triplets
        # else:
        predicted_encoding = self._storage[head].predicted_encoding
        predicted_mask = self._storage[head].predicted_mask
        next_skip_data_num = 0  # to skip the data pint for different goal in the same step
        pre_step_label = self._storage[head].step_label
        while True:
            next_skip_data_num += 1
            if self._storage[head + next_skip_data_num].step_label - pre_step_label == n:
                break

        next_obs = self._storage[head + next_skip_data_num].observation_item
        next_prev_action = self._storage[head + next_skip_data_num].prev_action_item
        next_goal_sentence = self._storage[head + next_skip_data_num].goal_item
        next_candidate = self._storage[head + next_skip_data_num].action_candidate_item

        # if self.supervised_rl_flag:
        #     next_graph_triplets = self._storage[head + next_skip_data_num].graph_triplets
        # else:
        next_predicted_encoding = self._storage[head + next_skip_data_num].predicted_encoding
        next_predicted_mask = self._storage[head + next_skip_data_num].predicted_mask

        tmp = next_final - head + 1 if self.accumulate_reward_from_final else n  # TODO: why n+1 in the origin code?

        rewards_up_to_next_final = [self.discount_gamma_game_reward ** i * self._storage[head + i].reward for i in
                                    range(tmp)]
        reward = torch.sum(torch.stack(rewards_up_to_next_final))

        graph_rewards_up_to_next_final = [self.discount_gamma_graph_reward ** i * self._storage[head + i].graph_reward
                                          for i in range(tmp)]
        graph_reward = torch.sum(torch.stack(graph_rewards_up_to_next_final))

        count_rewards_up_to_next_final = [self.discount_gamma_count_reward ** i * self._storage[head + i].count_reward
                                          for i in range(tmp)]
        count_reward = torch.sum(torch.stack(count_rewards_up_to_next_final))
        # if self.supervised_rl_flag:
        #     return (obs, prev_action, goal_sentence, candidate, chosen_indices, graph_triplets,
        #             reward + graph_reward + count_reward,
        #             next_obs, next_prev_action, next_goal_sentence, next_candidate, next_graph_triplets)
        # else:
        return (obs, prev_action, goal_sentence, candidate, chosen_indices,
                predicted_encoding, predicted_mask,
                reward + graph_reward + count_reward,
                next_obs, next_prev_action, next_goal_sentence, next_candidate,
                next_predicted_encoding, next_predicted_mask)

    def _encode_sample(self, idxes, ns):
        actual_indices, actual_ns = [], []
        # if self.supervised_rl_flag:
        #     obs, prev_action, goal_sentence, candidate, chosen_indices, graph_triplet_adjs, reward, \
        #     next_obs, next_prev_action, next_goal_sentence, next_candidate, next_graph_triplet_adjs = \
        #         [], [], [], [], [], [], [], [], [], [], [], []
        # else:
        obs, prev_action, goal_sentence, candidate, chosen_indices, predicted_encoding, predicted_mask, reward, \
        next_obs, next_prev_action, next_goal_sentence, next_candidate, next_predicted_encoding, next_predicted_mask = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i, n in zip(idxes, ns):
            t = self._get_single_transition(i, n)
            if t is None:
                continue
            actual_indices.append(i)
            actual_ns.append(n)
            obs.append(t[0])
            prev_action.append(t[1])
            goal_sentence.append(t[2])
            candidate.append(t[3])
            chosen_indices.append(t[4])
            # if self.supervised_rl_flag:
            #     graph_triplet_adjs.append(t[5])
            #     reward.append(t[6])
            #     next_obs.append(t[7])
            #     next_prev_action.append(t[8])
            #     next_goal_sentence.append(t[9])
            #     next_candidate.append(t[10])
            #     next_graph_triplet_adjs.append(t[11])
            # else:
            predicted_encoding.append(t[5])
            predicted_mask.append(t[6])
            reward.append(t[7])
            next_obs.append(t[8])
            next_prev_action.append(t[9])
            next_goal_sentence.append(t[10])
            next_candidate.append(t[11])
            next_predicted_encoding.append(t[12])
            next_predicted_mask.append(t[13])

        if len(actual_indices) == 0:
            return None
        chosen_indices = np.array(chosen_indices)  # batch
        reward = torch.stack(reward, 0)  # batch
        actual_ns = np.array(actual_ns)

        # if self.supervised_rl_flag:
        #     return [obs, prev_action, goal_sentence, candidate, chosen_indices, graph_triplet_adjs, reward,
        #             next_obs, next_prev_action, next_goal_sentence, next_candidate, next_graph_triplet_adjs,
        #             actual_indices, actual_ns]
        # else:
        return [obs, prev_action, goal_sentence, candidate, chosen_indices,
                predicted_encoding, predicted_mask,
                reward,
                next_obs, next_prev_action, next_goal_sentence, next_candidate,
                next_predicted_encoding, next_predicted_mask,
                actual_indices, actual_ns]

    def sample(self, batch_size, beta=0, multi_step=1):

        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        # sample n
        ns = np.random.randint(1, multi_step + 1, size=batch_size)
        encoded_sample = self._encode_sample(idxes, ns)
        if encoded_sample is None:
            return None
        actual_indices = encoded_sample[-2]
        for idx in actual_indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return encoded_sample + [weights]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            if priority > 0:
                assert 0 <= idx < len(self._storage)
                self._it_sum[idx] = priority ** self._alpha
                self._it_min[idx] = priority ** self._alpha
                self._max_priority = max(self._max_priority, priority)
            else:
                print("something wrong with priority: ", str(priority))
                return False
        return True

    def avg_rewards(self):
        if len(self._storage) == 0:
            return 0.0
        rewards = [self._storage[i].reward for i in range(len(self._storage))]
        return to_np(torch.mean(torch.stack(rewards)))


class BalancedTransitionMemory(object):

    def __init__(self, capacity, seed, supervised_rl_flag):
        self._storage = {0: [], 1: []}  # split the transitions with reward 1 and reward 0
        self.capacity = capacity
        self._next_idx = {0: 0, 1: 0}
        self.rng = np.random.RandomState(seed)
        self.supervised_rl_flag = supervised_rl_flag

    @property
    def storage(self):
        """content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """Max capacity of the buffer"""
        return self.capacity

    def length(self):
        return len(self._storage[0]) + len(self._storage[1])

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self._storage[0]) >= n_samples / 2 and len(self._storage[1]) >= n_samples / 2

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self._storage[0]) == self.buffer_size or len(self._storage[1]) == self.buffer_size

    def add(self, *args):
        """
        add a new transition to the buffer
        """
        if self.supervised_rl_flag:
            data = TransitionRecordSuperRL(*args)
        else:
            data = TransitionRecordUnSuperRL(*args)
        if data.next_reward.is_cuda:  # GPU
            storage_label = int(data.next_reward.data.cpu().item())
        else:
            storage_label = int(data.next_reward.data.item())
        if self._next_idx[storage_label] >= len(self._storage[storage_label]):
            self._storage[storage_label].append(data)
        else:
            self._storage[storage_label][self._next_idx[storage_label]] = data
        self._next_idx[storage_label] = (self._next_idx[storage_label] + 1) % self.capacity

    def sample(self, batch_size):
        """
        sample data for the transition and reward model updating
        """
        # o_t, h_t, g_t, a_t, r_t+1, o_t+1, h_t+1, to train p(r_t+1|z_t+1,a_t, g_t) and p(z_t+1|a_t, z_t)
        observation_sampled, graph_triplets_sampled, goal_sampled, \
        selected_action_sampled, next_reward_sampled, \
        next_observation_sampled, next_graph_triplets_sampled, = [], [], [], [], [], [], []

        for storage_label in self._storage.keys():
            indices = self.rng.choice(len(self._storage[storage_label]), int(batch_size / 2))

            for idx in indices:
                sampled_item = self._storage[storage_label][idx]
                observation_sampled.append(sampled_item[0])
                graph_triplets_sampled.append(sampled_item[1])
                goal_sampled.append(sampled_item[2])
                selected_action_sampled.append(sampled_item[3])
                next_reward_sampled.append(sampled_item[4])
                next_observation_sampled.append(sampled_item[5])
                next_graph_triplets_sampled.append(sampled_item[6])

        return [observation_sampled, graph_triplets_sampled, goal_sampled,
                selected_action_sampled, next_observation_sampled, next_graph_triplets_sampled,
                next_reward_sampled]
