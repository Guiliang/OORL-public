import copy
import math
import random
import torch.nn.functional as F
import numpy as np
import torch

from generic.data_utils import process_facts, get_goal_sentence, preproc, serialize_facts, adj_to_triplets, \
    generate_triplets_filter_mask, matching_object_from_obs, diff_triplets, check_action_repeat, \
    extract_goal_sentence_from_obs
from generic.model_utils import to_pt, to_np


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.action = None
        self.parent = None
        self.candidate_child = ['restart']
        self.child_N = np.zeros([len(self.candidate_child)], dtype=np.float32)
        self.child_W = np.zeros([len(self.candidate_child)], dtype=np.float32)

        self.ingredients = []
        self.h_hidden_reward = None
        self.goal_description = None
        self.reward = 0

        self.N = 0
        self.W = 0
        self.id = 0
        self.memory = np.zeros((1, 20, 99, 99), dtype="float32")

    # def revert_virtual_loss(self, up_to=None): pass
    #
    # def add_virtual_loss(self, up_to=None): pass
    #
    # def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None, discount_value=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self, id, memory, action, observation, reward, candidate_child, candidate_child_q_values, parent,
                 h_hidden_reward, hx_dynamic, cx_dynamic, goal_description, goal_sentence_store, ingredients, level,
                 random_seed, c_puct, discount_rate, done):
        self.id = id
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth + 1
        if parent is not None:
            self.parent = parent
        else:
            self.parent = DummyNode()
        self.action = action
        self.memory = memory
        self.reward = reward
        self.observation = observation
        self.candidate_child = candidate_child
        self.candidate_child_q_values = candidate_child_q_values
        self.children = {}
        if self.candidate_child is not None:
            self.child_N = np.zeros([len(self.candidate_child)], dtype=np.float32)
            self.child_W = np.zeros([len(self.candidate_child)], dtype=np.float32)
        else:
            self.child_N = None
            self.child_W = None

        self.is_expanded = False
        self.random_seed = random_seed
        self.level = level
        self.c_puct = c_puct
        self.discount_rate = discount_rate

        self.h_hidden_reward = None

        # These infos are for generating the next goal sentences
        self.ingredients = ingredients
        self.h_hidden_reward = h_hidden_reward
        self.hx_dynamic = hx_dynamic
        self.cx_dynamic = cx_dynamic
        self.goal_description = goal_description

        self.done = done
        self.goal_sentence_store = goal_sentence_store

    @property
    def N(self):
        action_idx = self.parent.candidate_child.index(self.action)
        return self.parent.child_N[action_idx]

    @N.setter
    def N(self, value):
        action_idx = self.parent.candidate_child.index(self.action)
        self.parent.child_N[action_idx] = value

    @property
    def W(self):
        action_idx = self.parent.candidate_child.index(self.action)
        return self.parent.child_W[action_idx]

    @W.setter
    def W(self, value):
        action_idx = self.parent.candidate_child.index(self.action)
        self.parent.child_W[action_idx] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def U(self):
        """
        Returns the current U of the node.
        """
        if self.action is not None:
            return self.c_puct * np.sqrt(math.log(self.parent.N + 1) / (1 + self.N))
        else:
            return float(0)

    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    def child_U(self):
        return self.c_puct * np.sqrt(math.log(1 + self.N) / (1 + self.child_N))

    def apply_q_values(self):
        all_candidate_child_q_values = []
        for goal in self.candidate_child_q_values.keys():
            all_candidate_child_q_values.append(self.candidate_child_q_values[goal])
        all_candidate_child_q_values = np.stack(all_candidate_child_q_values, axis=0)
        self.child_W = None

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.child_U()
        # max_indices = np.where(child_score == child_score.max())[0]  # might have multiple max values, then random from them
        # return random.choice(max_indices)
        return np.argmax(child_score)

    def select(self, max_search_depth):
        current_node = self
        tree_level = self.level
        action_selected = []
        new_flag = False
        done = False
        search_depth = 0
        while current_node.is_expanded and not done and search_depth < max_search_depth:
            if len(current_node.candidate_child) == 0:
                break
            tree_level += 1
            best_action_idx = current_node.best_action()
            action_selected.append(current_node.candidate_child[best_action_idx])
            current_node, new_flag, done = current_node.get_child(current_node.candidate_child[best_action_idx],
                                                                  tree_level)
            search_depth += 1
            # if not current_node.is_expanded:
            #     break
        return current_node, action_selected, new_flag

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.N += 1
            current.W += value
            current = current.parent

    def expand(self):
        self.is_expanded = True

    def get_child(self, action, level):
        if action in self.children.keys():
            return self.children[action], False, self.done
        else:
            self.children[action] = MCTSNode(id=None,
                                             memory=None,
                                             action=action,
                                             observation=None,
                                             reward=None,
                                             candidate_child=None,
                                             candidate_child_q_values=None,
                                             parent=self,
                                             h_hidden_reward=None,
                                             hx_dynamic=None,
                                             cx_dynamic=None,
                                             goal_description=None,
                                             goal_sentence_store=None,
                                             ingredients=None,
                                             level=level,
                                             random_seed=self.random_seed,
                                             c_puct=self.c_puct,
                                             discount_rate=self.discount_rate,
                                             done=None)
            return self.children[action], True, False

    def backup_value(self, value, up_to, discount_value):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value * discount_value
        self.N += 1
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to, discount_value * self.discount_rate)

    def print_tree(self, log_file, indent):
        child_score = np.round(self.child_Q() + self.child_U(), 5)
        child_info = list(zip(self.candidate_child, child_score.tolist()))
        indent_str = indent * "------"
        print(
            "{0} Node Id: {1} | action: {2} | obs: {3} | next_goal: {4} | vis_num: {5} | reward: {6} | child score: {7}"
                .format(indent_str, self.id, self.action, self.observation,
                        self.goal_description, self.N, self.reward, child_info),
            file=log_file, flush=True)
        for child_node in self.candidate_child:
            if child_node in self.children.keys():
                self.children[child_node].print_tree(log_file=log_file, indent=indent + 1)
        return


class MCTSPlanning:

    def __init__(self, TreeEnv, agent_planner, extractor, candidate_triplets,
                 random_seed, log_file, planning_action_log, planning_tree_log,
                 rule_based_extraction, difficulty_level, debug_mode):
        # self.TreeEnv = TreeEnv
        self.rule_based_extraction = rule_based_extraction
        self.debug_mode = debug_mode
        self.random_move_prob_add = agent_planner.random_move_prob_add
        # self.max_scores = agent_planner.max_scores
        self.discount_rate = agent_planner.discount_rate
        self.TreeEnv = TreeEnv.copy()
        self._TreeEnv = TreeEnv.copy()  # to get candidate action
        self.log_file = log_file
        self.planning_action_log = planning_action_log
        self.planning_tree_log = planning_tree_log
        self.c_puct = agent_planner.c_puct
        self.agent_planner = agent_planner
        self.extractor = extractor
        self.random_seed = random_seed
        self.simulations_num = agent_planner.simulations_num
        self.moved_actions = ['restart']
        self.moved_goals = []
        self.moved_last_facts = []
        self.Id_count = 1
        self.current_scores = 0
        filter_mask = generate_triplets_filter_mask(triplet_set=candidate_triplets,
                                                    node2id=self.agent_planner.node2id,
                                                    relation2id=self.agent_planner.relation2id)
        self.filter_mask = filter_mask
        np.random.seed(self.random_seed)
        self.TreeEnv.seed(self.random_seed)
        self.difficulty_level = difficulty_level
        self.initialize_search()
        self.max_search_depth = agent_planner.max_search_depth

    def extract_info_from_observation(self, input_adj, previous_actions, current_observations, reward, threshold=0.5):
        with torch.no_grad():
            if self.extractor is not None:
                predicted_encodings, _, _, node_mask, _, _, _, relation_encodings, _, _ = \
                    self.extractor.compute_updated_dynamics(input_adj_m=input_adj,
                                                            actions=previous_actions,
                                                            observations=current_observations,
                                                            hx=None,
                                                            cx=None)
                model_output_adj = self.extractor.model.decode_graph(predicted_encodings, relation_encodings)
            else:
                if 'unsupervised' in self.agent_planner.task:
                    reward = to_pt(np.asarray(reward), enable_cuda=self.agent_planner.use_cuda, type='float')
                else:
                    reward = None
                predicted_encodings_prior, hx_new_prior, cx_new_prior, \
                predicted_encodings_post, hx_new_post, cx_new_prost, \
                node_mask, input_node_name, input_relation_name, node_encodings, relation_encodings, _, _ = \
                    self.agent_planner.compute_updated_dynamics(input_adj_m=input_adj,
                                                                actions=previous_actions,
                                                                observations=current_observations,
                                                                rewards=reward,
                                                                hx=None,
                                                                cx=None)
                model_output_adj = self.agent_planner.model.decode_graph(predicted_encodings_post, relation_encodings)
                predicted_encodings = predicted_encodings_post
            if 'unsupervised' in self.agent_planner.task:
                pred_extract_adj_matrix = model_output_adj
            else:
                model_output_adj = (to_np(model_output_adj) > threshold).astype(int)
                model_output_adj = self.filter_mask * model_output_adj
                triplets_pred = adj_to_triplets(adj_matrix=model_output_adj,
                                                node_vocab=self.agent_planner.node_vocab,
                                                relation_vocab=self.agent_planner.relation_vocab)
                triplets_input = adj_to_triplets(adj_matrix=input_adj,
                                                 node_vocab=self.agent_planner.node_vocab,
                                                 relation_vocab=self.agent_planner.relation_vocab)
                triplets_pred = matching_object_from_obs(observations=current_observations,
                                                         actions=previous_actions,
                                                         node_vocab=self.agent_planner.node_vocab,
                                                         pred_triplets=triplets_pred,
                                                         input_triplets=triplets_input)
                pred_extract_adj_matrix = self.agent_planner.get_graph_adjacency_matrix(triplets_pred)

        return pred_extract_adj_matrix, predicted_encodings, node_mask

    def planner_prediction(self, input_adj, previous_actions,
                           previous_goal_sentences,
                           current_goal_sentences,
                           action_candidate_list,
                           prev_h_goal_dict,
                           filter_mask):
        """planner interact with the RL model"""
        with torch.no_grad():
            """predict the next latent state"""
            if 'unsupervised' in self.agent_planner.task:
                reward = to_pt(np.asarray([0]), enable_cuda=self.agent_planner.use_cuda, type='float')  # pad rewards
            else:
                reward = None
            predicted_encodings_prior, hx_new_prior, cx_new_prior, \
            predicted_encodings_post, hx_new_post, cx_new_prost, \
            node_mask, input_node_name, input_relation_name, node_embeddings, relation_embeddings, \
            _, _ = \
                self.agent_planner.compute_updated_dynamics(input_adj_m=input_adj,
                                                            actions=previous_actions,
                                                            observations=['<pad>'],
                                                            rewards=reward,
                                                            hx=None,
                                                            cx=None)
            goal_encodings_sequences, goal_mask = None, None
            predicted_encodings = predicted_encodings_prior

            pred_planning_adj = self.agent_planner.model.decode_graph(predicted_encodings, relation_embeddings)
            if 'unsupervised' in self.agent_planner.task:
                pred_planner_adj_matrix = pred_planning_adj
            else:
                filter_mask = np.repeat(filter_mask, len(input_adj), axis=0)
                adj_matrix = (to_np(pred_planning_adj) > 0.5).astype(int)
                pred_planner_adj_matrix = filter_mask * adj_matrix

            """predict the next reward"""
            reward_goal_dict = {}
            current_h_goal_dict = {}

            if 'unsupervised' in self.agent_planner.task and 'goal' in self.agent_planner.task:
                input_actions = self.agent_planner.get_word_input(previous_actions,
                                                                  minimum_len=10)  # batch x action_len
                action_encodings_sequences, action_mask = \
                    self.agent_planner.model.encode_text_for_reward_prediction(input_actions)

                for previous_goal_sentence in previous_goal_sentences:
                    input_goals = self.agent_planner.get_word_input([previous_goal_sentence],
                                                                    minimum_len=20)  # batch x goal_len
                    goal_encodings_sequences, goal_mask = \
                        self.agent_planner.model.encode_text_for_reward_prediction(input_goals)
                    pred_0_rewards = self.agent_planner.compute_rewards_unsupervised(
                        predicted_encodings=predicted_encodings,
                        node_mask=node_mask,
                        action_encodings_sequences=action_encodings_sequences,
                        action_mask=action_mask,
                        goal_encodings_sequences=goal_encodings_sequences,
                        goal_mask=goal_mask)
                    pred_0_rewards = F.softmax(pred_0_rewards, dim=1)
                    if self.agent_planner.use_cuda:
                        rewards = pred_0_rewards.detach().cpu().numpy()
                    else:
                        rewards = pred_0_rewards.numpy()
                    reward = 1 - np.argmax(rewards[0])  # [pos_prob, neg_prob]
                    reward_goal_dict.update({previous_goal_sentence: reward})
            elif 'unsupervised' in self.agent_planner.task and 'goal' not in self.agent_planner.task:
                input_actions = self.agent_planner.get_word_input(previous_actions,
                                                                  minimum_len=10)  # batch x action_len
                action_encodings_sequences, action_mask = self.agent_planner.model.encode_dynamic_action(input_actions)
                pred_0_rewards = \
                    self.agent_planner.compute_rewards_unsupervised(predicted_encodings=predicted_encodings,
                                                                    node_mask=node_mask,
                                                                    action_encodings_sequences=action_encodings_sequences,
                                                                    action_mask=action_mask,
                                                                    goal_encodings_sequences=goal_encodings_sequences,
                                                                    goal_mask=goal_mask)
                pred_0_rewards = F.softmax(pred_0_rewards, dim=1)
                if self.agent_planner.use_cuda:
                    rewards = pred_0_rewards.detach().cpu().numpy()
                else:
                    rewards = pred_0_rewards.numpy()
                reward = 1 - np.argmax(rewards[0])  # [pos_prob, neg_prob]
                reward_goal_dict.update({'Goal': reward})
            else:
                for previous_goal_sentence in previous_goal_sentences:
                    if self.agent_planner.model.reward_predictor_apply_rnn:  # apply rnn
                        if previous_goal_sentence in prev_h_goal_dict.keys():
                            prev_h = prev_h_goal_dict[previous_goal_sentence].unsqueeze(0)
                        else:  # no rnn
                            prev_h = None
                    else:
                        prev_h = None
                    pred_0_rewards, rnn_output = \
                        self.agent_planner.compute_rewards(node_encodings=predicted_encodings,
                                                           node_mask=node_mask,
                                                           previous_actions=previous_actions,
                                                           goal_sentences=[previous_goal_sentence],
                                                           if_apply_rnn=self.agent_planner.model.reward_predictor_apply_rnn,
                                                           h_t_minus_one=prev_h,
                                                           )
                    if self.agent_planner.use_cuda:
                        rewards = pred_0_rewards.detach().cpu().numpy()
                    else:
                        rewards = pred_0_rewards.numpy()
                    reward = 1 - np.argmax(rewards[0])  # [pos_prob, neg_prob]
                    reward_goal_dict.update({previous_goal_sentence: reward})

                    if self.agent_planner.model.reward_predictor_apply_rnn:  # apply rnn
                        curr_h = rnn_output[0]
                        current_h_goal_dict.update({previous_goal_sentence: curr_h})
                    else:  # no rnn
                        current_h_goal_dict = {}

                    if previous_goal_sentence == 'find kitchen' and self.difficulty_level == 5:
                        reward_goal_dict.update({previous_goal_sentence: 0})

                """predict the next Q values"""
            next_q_values_goal_dict = {}
            if 'unsupervised' in self.agent_planner.task and 'goal' not in self.agent_planner.task:
                pass
            else:
                if len(action_candidate_list) > 0:
                    for goal_sentence in current_goal_sentences:
                        next_input_goal_ids = self.agent_planner.get_word_input([goal_sentence], minimum_len=20)
                        next_input_candidate_word_ids = \
                            self.agent_planner.get_action_candidate_list_input([action_candidate_list])

                        pred_next_candidate_q_values, cand_mask = \
                            self.agent_planner.compute_q_values_multi_candidates(node_encodings=predicted_encodings,
                                                                                 node_mask=node_mask,
                                                                                 input_candidate_word_ids=next_input_candidate_word_ids,
                                                                                 input_goals_ids=next_input_goal_ids)
                        if self.agent_planner.use_cuda:
                            q_values = pred_next_candidate_q_values.squeeze(0).detach().cpu().numpy()
                        else:
                            q_values = pred_next_candidate_q_values.squeeze(0).detach().numpy()
                        next_q_values_goal_dict.update({goal_sentence: q_values})

        return pred_planner_adj_matrix, reward_goal_dict, next_q_values_goal_dict, current_h_goal_dict

    def initialize_search(self):
        # self.TreeEnv.reset()  # start the game
        # filter look and examine actions
        # commands_ = infos["admissible_commands"]
        # for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
        #     commands_.remove(cmd_)

        action_candidate_list, observation_string, origin_observation_string, facts_seen, scores, done = \
            self.interact(action='restart',
                          last_facts=set(),
                          action_selected_all=self.moved_actions,
                          repeat_flag=False if self.difficulty_level == 9 and 'examine cookbook' in self.moved_actions else True
        )
        triplets_real = [sorted(serialize_facts(facts_seen))]
        self.moved_last_facts.append(facts_seen)

        # observation_string, action_candidate_list = self.agent_planner.get_game_info_at_certain_step_lite(obs, infos)
        # if " your score has just gone up by one point ." in observation_string:
        #     observation_string = observation_string.replace(" your score has just gone up by one point .", "")
        # _last_facts = set()
        # _facts_seen = process_facts(_last_facts,
        #                             infos["game"],
        #                             infos["facts"],
        #                             infos["last_action"],
        #                             'restart')
        # triplets_real = [sorted(serialize_facts(_facts_seen))]
        # _state = self.agent.get_graph_adjacency_matrix([sorted(serialize_facts(_facts_seen))])

        _input_adj_init = np.zeros((1,
                                    len(self.agent_planner.relation_vocab),
                                    len(self.agent_planner.node_vocab),
                                    len(self.agent_planner.node_vocab)),
                                   dtype="float32")

        current_0_observations = [observation_string]
        previous_0_actions = [preproc('restart', tokenizer=self.agent_planner.nlp)]
        reward = [0]
        self.current_scores = scores
        pred_extract_adj_matrix, predicted_encodings, node_mask = \
            self.extract_info_from_observation(input_adj=_input_adj_init,
                                               previous_actions=previous_0_actions,
                                               current_observations=current_0_observations,
                                               reward=reward)

        if 'unsupervised' in self.agent_planner.task and 'goal' in self.agent_planner.task:
            prev_h_goal_dict = {}
            goal_sentences_init, ingredients_init, goal_sentence_store_init = \
                extract_goal_sentence_from_obs(
                    agent=self.agent_planner,
                    object_encodings=predicted_encodings,
                    object_mask=node_mask,
                    pre_goal_sentence=None,
                    obs=None,
                    obs_origin=None,
                    ingredients=None,
                    goal_sentence_store=None,
                    difficulty_level=self.difficulty_level,
                    rule_based_extraction=self.rule_based_extraction
                )
            goal_sentences_next, ingredients, goal_sentence_store = \
                extract_goal_sentence_from_obs(
                    agent=self.agent_planner,
                    object_encodings=predicted_encodings,
                    object_mask=node_mask,
                    pre_goal_sentence=goal_sentences_init,
                    obs=observation_string,
                    obs_origin=origin_observation_string,
                    ingredients=copy.copy(ingredients_init),
                    goal_sentence_store=goal_sentence_store_init,
                    difficulty_level=self.difficulty_level,
                    rule_based_extraction=self.rule_based_extraction
                )
        elif 'unsupervised' in self.agent_planner.task and 'goal' in self.agent_planner.task:
            goal_sentences_init = None
            goal_sentences_next = None
            prev_h_goal_dict = None
        else:
            prev_h_goal_dict = {}
            goal_sentences_init, ingredients_init, goal_sentence_store_init = \
                get_goal_sentence(pre_goal_sentence=None,
                                  ingredients=set(),
                                  difficulty_level=self.difficulty_level,
                                  recheck_ingredients=False)
            # if self.difficulty_level == 9:
            #     goal_sentence = ' * '.join(goal_sentences_init)

            goal_sentences_next, ingredients, goal_sentence_store = \
                get_goal_sentence(pre_goal_sentence=goal_sentences_init,
                                  obs=current_0_observations[0],
                                  state=pred_extract_adj_matrix,
                                  ingredients=copy.copy(ingredients_init),
                                  node2id=self.agent_planner.node2id,
                                  relation2id=self.agent_planner.relation2id,
                                  node_vocab=self.agent_planner.node_vocab,
                                  relation_vocab=self.agent_planner.relation_vocab,
                                  goal_sentence_store=[],
                                  difficulty_level=self.difficulty_level,
                                  recheck_ingredients=False)
            # tmp_diff, num_redundant, num_lack = diff_triplets(triplets1=triplets_pred[0],
            #                                                   triplets2=triplets_real[0])
        self.moved_goals.append(goal_sentences_init)

        pred_planner_adj_matrix, reward_dict, candidate_q_values_dict, curr_h_dict = \
            self.planner_prediction(input_adj=_input_adj_init,
                                    previous_actions=previous_0_actions,
                                    previous_goal_sentences=goal_sentences_init,
                                    current_goal_sentences=goal_sentences_next,
                                    action_candidate_list=action_candidate_list,
                                    prev_h_goal_dict=prev_h_goal_dict,
                                    filter_mask=self.filter_mask)

        # if self.difficulty_level == 9:
        #     goal_sentence_next = ' * '.join(goal_sentences_next)

        self.root = MCTSNode(id=self.Id_count,
                             memory=pred_extract_adj_matrix,
                             action='restart',
                             observation=observation_string,
                             reward=reward_dict,
                             candidate_child=action_candidate_list,
                             candidate_child_q_values=candidate_q_values_dict,
                             parent=None,
                             h_hidden_reward=curr_h_dict,
                             hx_dynamic=None,
                             cx_dynamic=None,
                             goal_description=goal_sentences_next,
                             goal_sentence_store=[],
                             ingredients=set(),
                             level=1,
                             random_seed=self.random_seed,
                             c_puct=self.c_puct,
                             discount_rate=self.discount_rate,
                             done=False, )
        self.Id_count += 1
        self.root.expand()

    def interact(self, action, last_facts, action_selected_all, repeat_flag=True):
        if action == 'restart':
            obs, infos = self.TreeEnv.reset()
            done = False
            scores = 0
        else:
            obs, scores, done, infos = self.TreeEnv.step(action)
        origin_observation_string = copy.copy(obs)
        observation_string, action_candidate_list = self.agent_planner.get_game_info_at_certain_step_lite(obs, infos)
        if " your score has just gone up by one point ." in observation_string:
            observation_string = observation_string.replace(" your score has just gone up by one point .", "")

        facts_seen = process_facts(last_facts, infos["game"], infos["facts"], infos["last_action"], action)

        if self.difficulty_level == 5:
            action_candidate_list_copy = copy.copy(action_candidate_list)
            for action in action_candidate_list_copy:
                if 'close' in action:
                    action_candidate_list.remove(action)

        if not repeat_flag:
            action_candidate_list_copy = copy.copy(action_candidate_list)
            for action in action_candidate_list_copy:
                if action in action_selected_all: #  or 'eat' in action:
                    action_candidate_list.remove(action)

        return action_candidate_list, observation_string, origin_observation_string, facts_seen, scores, done

    def get_candidate_actions(self, action_selected_all, repeat_flag=True):
        """
        we assume we know the candidate actions
        """
        _obs, _infos = self._TreeEnv.reset()
        _cmd = action_selected_all[0]
        _facts_seen = process_facts(set(), _infos["game"], _infos["facts"], _infos["last_action"], _cmd)
        _last_facts = _facts_seen
        _done = False
        for _cmd in action_selected_all[1:]:  # ignore restart, since restart = env.reset()
            _obs, _scores, _done, _infos = self._TreeEnv.step(_cmd)
            _facts_seen = process_facts(_last_facts, _infos["game"], _infos["facts"], _infos["last_action"], _cmd)
            _last_facts = _facts_seen

        observation_string, action_candidate_list = self.agent_planner.get_game_info_at_certain_step_lite(_obs, _infos)
        if " your score has just gone up by one point ." in observation_string:
            observation_string = observation_string.replace(" your score has just gone up by one point .", "")

        if self.difficulty_level == 5:
            action_candidate_list_copy = copy.copy(action_candidate_list)
            for action in action_candidate_list_copy:
                if 'close' in action:
                    action_candidate_list.remove(action)

        if not repeat_flag:
            action_candidate_list_copy = copy.copy(action_candidate_list)
            for action in action_candidate_list_copy:
                if action in action_selected_all or 'eat' in action:
                    action_candidate_list.remove(action)

        if _done and self.difficulty_level is not 9:
            action_candidate_list = []

        triplet_ground_truth = serialize_facts(_facts_seen)
        return action_candidate_list, triplet_ground_truth

    def add_info_backup(self, expanded_node, action_selected_all, new_flag):
        if new_flag:
            # if self.Id_count == 5:
            #     print('debug')
            action_candidate_list, _ = \
                self.get_candidate_actions(action_selected_all=action_selected_all,
                                           repeat_flag=False if self.difficulty_level == 9 and 'examine cookbook' in action_selected_all else True)
            # repeat_flag=False if 'unsupervised' in self.agent_planner.task else True)
            previous_t_actions = [preproc(expanded_node.action, tokenizer=self.agent_planner.nlp)]
            input_t_adj = expanded_node.parent.memory

            previous_t_goal_sentences = expanded_node.parent.goal_description
            t_goal_sentences = expanded_node.parent.goal_description
            # if self.difficulty_level == 9:
            #     tmp = ' * '.join(expanded_node.parent.goal_description)
            #     current_t_goal_sentences = [' * '.join(expanded_node.parent.goal_description)]
            pred_planner_adj_matrix, \
            reward_dict, candidate_q_values_dict, \
            curr_h_dict = self.planner_prediction(input_adj=input_t_adj,
                                                  previous_actions=previous_t_actions,
                                                  previous_goal_sentences=previous_t_goal_sentences,
                                                  current_goal_sentences=t_goal_sentences,
                                                  action_candidate_list=action_candidate_list,
                                                  prev_h_goal_dict=expanded_node.parent.h_hidden_reward,
                                                  filter_mask=self.filter_mask)

            # predicted_encodings, _, _, attn_mask, _, _, _, relation_encodings = \
            #     self.agent.compute_updated_dynamics(input_adj_m=input_t_adj,
            #                                         actions=previous_t_actions,
            #                                         observations=current_t_observations,
            #                                         hx=None,
            #                                         cx=None)
            #
            # model_output_adj = self.agent.model.decode_graph(predicted_encodings, relation_encodings)
            # model_output_adj = (to_np(model_output_adj) > 0.5).astype(int)
            # model_output_adj = self.filter_mask * model_output_adj
            #
            # triplets_pred = adj_to_triplets(adj_matrix=model_output_adj,
            #                                 node_vocab=self.agent.node_vocab,
            #                                 relation_vocab=self.agent.relation_vocab)
            # triplets_input = adj_to_triplets(adj_matrix=input_t_adj,
            #                                  node_vocab=self.agent.node_vocab,
            #                                  relation_vocab=self.agent.relation_vocab)
            #
            # triplets_pred = matching_object_from_obs(observations=current_t_observations,
            #                                          actions=previous_t_actions,
            #                                          node_vocab=self.agent.node_vocab,
            #                                          pred_triplets=triplets_pred,
            #                                          input_triplets=triplets_input)
            # pred_adj_matrix = self.agent.get_graph_adjacency_matrix(triplets_pred)
            #
            # tmp_diff = diff_triplets(triplets1=triplet_ground_truth, triplets2=triplets_pred[0])
            #
            # if self.Id_count == 102:
            #     print(tmp_diff)
            #     print("debug")
            #
            # goal_sentence_next, ingredients, goal_sentence_store = \
            #     get_goal_sentence(pre_goal_sentence=expanded_node.parent.goal_description,
            #                       obs=observation_string,
            #                       state=pred_adj_matrix,
            #                       ingredients=copy.copy(expanded_node.parent.ingredients),
            #                       node2id=self.agent.node2id,
            #                       relation2id=self.agent.relation2id,
            #                       node_vocab=self.agent.node_vocab,
            #                       relation_vocab=self.agent.relation_vocab,
            #                       goal_sentence_store=expanded_node.parent.goal_sentence_store)
            #
            # goal_sentence = expanded_node.parent.goal_description
            # current_t_goal_sentences = [preproc(goal_sentence, tokenizer=self.agent.nlp)]
            # current_t_observations = [observation_string]
            #
            # prev_h = expanded_node.parent.h_hidden_reward.unsqueeze(0)
            # pred_t_rewards, rnn_t_output = self.agent.compute_rewards(node_encodings=predicted_encodings,
            #                                                           node_mask=attn_mask,
            #                                                           current_observations=current_t_observations,
            #                                                           previous_actions=previous_t_actions,
            #                                                           current_goal_sentences=current_t_goal_sentences,
            #                                                           if_apply_rnn=self.agent.model.reward_predictor_apply_rnn,
            #                                                           h_t_minus_one=prev_h,
            #                                                           )

            # # get new graph from dynamic model
            # predicted_encodings, hx_new_mu, hx_new_logvar, cx_new, attn_mask, \
            # input_node_name, input_relation_name, node_encodings, relation_encodings = \
            #     self.agent.compute_updated_dynamics(input_adj_m=previous_t_state,
            #                                         action=previous_t_actions,
            #                                         observations=current_t_observations,
            #                                         hx=None,
            #                                         cx=None)
            #
            # predict_output_adj = self.agent.model.decode_graph(predicted_encodings, relation_encodings)
            #
            # state = (to_np(predict_output_adj) > 0.5).astype(int)
            #
            # goal_sentence_next, ingredients, goal_sentence_store = \
            #     get_goal_sentence(pre_goal_sentence=expanded_node.parent.goal_description,
            #                       obs=observation_strings,
            #                       state=state,
            #                       ingredients=expanded_node.parent.ingredients,
            #                       node2id=self.agent.node2id,
            #                       relation2id=self.agent.relation2id,
            #                       goal_sentence_store=expanded_node.parent.goal_sentence_store)
            #
            # goal_sentence = expanded_node.parent.goal_description
            # current_t_adjacency_matrix = state
            # current_t_goal_sentences = [preproc(goal_sentence, tokenizer=self.agent.nlp)]
            # prev_h = expanded_node.parent.h_hidden_reward.unsqueeze(0)
            # # get new rewards from reward prediction model
            #
            # adjacency_matrix = to_pt(current_t_adjacency_matrix, self.agent.use_cuda, type='float')
            # input_node_name = self.agent.get_graph_node_name_input()
            # input_relation_name = self.agent.get_graph_relation_name_input()
            # node_encodings, _, _, node_mask = self.agent.model.encode_graph(input_node_name,
            #                                                                 input_relation_name,
            #                                                                 adjacency_matrix)
            # pred_t_rewards, rnn_t_output = \
            #     self.agent.compute_rewards(node_encodings=node_encodings,
            #                                node_mask=node_mask,
            #                                current_observations=current_t_observations,
            #                                previous_actions=previous_t_actions,
            #                                current_goal_sentences=current_t_goal_sentences,
            #                                if_apply_rnn=self.agent.model.reward_predictor_apply_rnn,
            #                                h_t_minus_one=prev_h)

            # if self.agent.use_cuda:
            #     rewards = pred_t_rewards.detach().cpu().numpy()
            # else:
            #     rewards = pred_t_rewards.numpy()
            # reward = 1 - np.argmax(rewards[0])  # [pos_prob, neg_prob]
            #
            # # if action_selected_all[-1] == 'eat meal' and 'prepare meal' in action_selected_all:
            # #     reward = 1  # hard winning rule for textworld
            # # print("debug")
            #
            # curr_h = rnn_t_output[0]

            expanded_node.memory = pred_planner_adj_matrix
            expanded_node.action = action_selected_all[-1]
            expanded_node.observation = ['<pad>']  # obs is unknown during planning
            expanded_node.candidate_child = action_candidate_list
            expanded_node.candidate_child_q_values = candidate_q_values_dict
            expanded_node.h_hidden_reward = curr_h_dict
            expanded_node.hx_dynamic = None
            expanded_node.cx_dynamic = None
            expanded_node.reward = reward_dict
            expanded_node.child_N = np.zeros([len(action_candidate_list)], dtype=np.float32)
            expanded_node.child_W = np.zeros([len(action_candidate_list)], dtype=np.float32)
            expanded_node.goal_description = expanded_node.parent.goal_description
            expanded_node.ingredients = expanded_node.parent.ingredients
            expanded_node.id = self.Id_count
            expanded_node.done = False
            expanded_node.goal_sentence_store = expanded_node.parent.goal_sentence_store
            self.Id_count += 1
            expanded_node.expand()

        # back up
        if len(expanded_node.reward.values()) > 0:
            backup_rewards = max(expanded_node.reward.values())
        else:
            backup_rewards = -1  # goal is not correctly extracted, a bug detected, return a -1 reward
        expanded_node.backup_value(backup_rewards, self.root, self.discount_rate)

        return expanded_node

    def move(self, extract_threshold=0.5):
        select_probs = self.root.child_N / np.sum(self.root.child_N)
        select_action_probs = ["{0}:{1}".format(self.root.candidate_child[idx], select_probs[idx]) for idx in
                               range(len(select_probs))]

        moved_action = None
        for action_idx in range(len(self.root.candidate_child)):
            moved_action = self.root.candidate_child[action_idx]

            rewards = self.root.children[moved_action].reward.values()
            if len(rewards) > 0:
                reward = max(rewards)
            else:
                continue
            if reward == 1:
                print("Reward select from {0}".format(select_action_probs),
                      file=self.planning_action_log,
                      flush=True)
                print("Reward select from {0}".format(select_action_probs),
                      file=self.planning_tree_log,
                      flush=True)
                break
            else:
                moved_action = None
        if moved_action is None:
            if np.max(select_probs) <= (1 / len(self.root.child_N) + self.random_move_prob_add):
                print("random select from {0}".format(select_action_probs),
                      file=self.planning_action_log,
                      flush=True)
                print("random select from {0}".format(select_action_probs),
                      file=self.planning_tree_log,
                      flush=True)
                if self.difficulty_level == 5:
                    while True:
                        moved_action_idx = \
                            np.random.choice(np.arange(0, len(self.root.child_N)),
                                                            p=self.root.child_N / np.sum(self.root.child_N))
                        moved_action = self.root.candidate_child[moved_action_idx]
                        if 'go' in moved_action or ('open' in moved_action and 'door' in moved_action):
                            break
                else:
                    moved_action_idx = np.random.choice(np.arange(0, len(self.root.child_N)),
                                                        p=self.root.child_N / np.sum(self.root.child_N))
                    moved_action = self.root.candidate_child[moved_action_idx]

            else:
                print("Argmax select from {0}".format(select_action_probs),
                      file=self.planning_action_log,
                      flush=True)
                print("Argmax select from {0}".format(select_action_probs),
                      file=self.planning_tree_log,
                      flush=True)
                moved_action_idx = np.argmax(select_probs)
                moved_action = self.root.candidate_child[moved_action_idx]

        if 'open fridge' in self.root.candidate_child and self.root.goal_description[
            0] != 'check the cookbook in the kitchen for the recipe':
            moved_action = 'open fridge'
        # else:

        # if moved_action == 'cook banana with oven':
        # extract_threshold = 0.99
        # print("debug")

        self.moved_actions += [moved_action]
        self.moved_goals += [self.root.goal_description]
        print(moved_action, self.root.goal_description)
        # if moved_action == 'examine cookbook':
        #     print('debugging')

        action_candidate_list, observation_string, origin_observation_string, facts_seen, scores, done = \
            self.interact(action=moved_action,
                          last_facts=self.moved_last_facts[-1],
                          action_selected_all=self.moved_actions,
                          repeat_flag=False if self.difficulty_level == 9 and 'examine cookbook' in self.moved_actions else True
                          )  # interact with env tgo get obs
        if len(action_candidate_list) == 0:
            return scores, False, observation_string

        # for action in self.moved_actions:  # remove repeated actions
        #     if action in action_candidate_list:
        #         action_candidate_list.remove(action)
        self.moved_last_facts.append(facts_seen)
        selected_node = self.root.children[moved_action]

        if len(selected_node.reward.values()) > 0:
            reward = max(selected_node.reward.values())
        else:
            reward = -1  # goal is not correctly extracted, a bug detected, stop the planning
            return scores, False, observation_string

        self.current_scores = scores
        pred_extract_adj_matrix, predicted_encodings, node_mask = self.extract_info_from_observation(
            input_adj=selected_node.parent.memory,
            previous_actions=[moved_action],
            current_observations=[observation_string],
            threshold=extract_threshold,
            reward=reward,
        )
        next_q_values_goal_dict = {}  # recompute action values based on the extracted states and new goals
        if 'unsupervised' in self.agent_planner.task and 'goal' not in self.agent_planner.task:
            ingredients = None
            goal_sentences_next = None
            goal_sentence_store = None
            action_candidate_list_copy = copy.copy(action_candidate_list)
            # for action in action_candidate_list_copy:
            #     if action in moved_action or 'eat' in action:
            #         action_candidate_list.remove(action)
            # if 'open fridge' in self.root.candidate_child and self.root.goal_description[
            #     0] != 'check the cookbook in the kitchen for the recipe':
            #     moved_action = 'open fridge'
            # else:
        elif 'unsupervised' in self.agent_planner.task and 'goal' in self.agent_planner.task:
            goal_sentences_next, ingredients, goal_sentence_store = \
                extract_goal_sentence_from_obs(agent=self.agent_planner,
                                               object_encodings=predicted_encodings,
                                               object_mask=node_mask,
                                               pre_goal_sentence=selected_node.parent.goal_description,
                                               obs=observation_string,
                                               obs_origin=origin_observation_string,
                                               ingredients=selected_node.parent.ingredients,
                                               goal_sentence_store=selected_node.parent.goal_sentence_store,
                                               difficulty_level=self.difficulty_level,
                                               rule_based_extraction=self.rule_based_extraction)
        else:
            goal_sentences_next, ingredients, goal_sentence_store = \
                get_goal_sentence(pre_goal_sentence=selected_node.parent.goal_description,
                                  obs=observation_string,
                                  state=pred_extract_adj_matrix,
                                  ingredients=selected_node.parent.ingredients,
                                  node2id=self.agent_planner.node2id,
                                  relation2id=self.agent_planner.relation2id,
                                  node_vocab=self.agent_planner.node_vocab,
                                  relation_vocab=self.agent_planner.relation_vocab,
                                  goal_sentence_store=selected_node.parent.goal_sentence_store,
                                  difficulty_level=self.difficulty_level,
                                  recheck_ingredients=False)

        if goal_sentences_next is not None:
            for goal_sentence in goal_sentences_next:
                next_input_goal_ids = self.agent_planner.get_word_input([goal_sentence], minimum_len=20)
                next_input_candidate_word_ids = self.agent_planner.get_action_candidate_list_input(
                    [action_candidate_list])
                pred_next_candidate_q_values, cand_mask = \
                    self.agent_planner.compute_q_values_multi_candidates(node_encodings=predicted_encodings,
                                                                         node_mask=node_mask,
                                                                         input_candidate_word_ids=next_input_candidate_word_ids,
                                                                         input_goals_ids=next_input_goal_ids)
                if self.agent_planner.use_cuda:
                    q_values = pred_next_candidate_q_values.squeeze(0).detach().cpu().numpy()
                else:
                    q_values = pred_next_candidate_q_values.squeeze(0).detach().numpy()
                next_q_values_goal_dict.update({goal_sentence: q_values})

        self.root = MCTSNode(id=selected_node.id,
                             memory=pred_extract_adj_matrix,
                             action=moved_action,
                             observation=observation_string,
                             reward=selected_node.reward,
                             candidate_child=action_candidate_list,
                             candidate_child_q_values=next_q_values_goal_dict,
                             parent=self.root,
                             h_hidden_reward=selected_node.h_hidden_reward,
                             hx_dynamic=None,
                             cx_dynamic=None,
                             goal_description=goal_sentences_next,
                             goal_sentence_store=goal_sentence_store,
                             ingredients=ingredients,
                             level=selected_node.level,
                             random_seed=self.random_seed,
                             c_puct=self.c_puct,
                             discount_rate=self.discount_rate,
                             done=done, )
        self.root.expand()

        # if len(self.root.ingredients) == 0 and 'examine cookbook' in self.moved_actions and self.agent_planner.difficulty_level == 9:
        #     print("modifying c_puct for debug")
        #     self.c_puct = 10
        #     self.root.c_puct = self.c_puct

        return scores, True, observation_string

    def plan(self):
        while not self.root.done:
            for i in range(self.simulations_num):
                expanded_node, action_selected, new_flag = self.root.select(self.max_search_depth)
                expanded_node = self.add_info_backup(expanded_node=expanded_node,
                                                     action_selected_all=self.moved_actions + action_selected,
                                                     new_flag=new_flag)
            if self.debug_mode:
                self.root.print_tree(log_file=self.planning_tree_log, indent=0)
                print("\n\n", file=self.planning_tree_log, flush=True)
            scores, if_success, observation_string = self.move()
            # print(self.moved_actions, file=self.planning_tree_log, flush=True)
            # print("*" * 20 + " scores: {0}".format(scores) + "*" * 20 + "\n", file=self.planning_tree_log, flush=True)

            print('Obs: {0}'.format(observation_string), file=self.planning_action_log, flush=True)
            print(["{0} ({1})".format(list(pair)[0], list(pair)[1]) for pair in
                   zip(self.moved_actions, self.moved_goals)], file=self.planning_action_log, flush=True)
            print("*" * 20 + " scores: {0}".format(scores) + "*" * 20 + "\n", file=self.planning_action_log, flush=True)

            # if len(self.moved_actions) > self.agent_planner.eval_max_nb_steps_per_episode:  # too many actions, end the game
            #     break
            if len(self.moved_actions) > 50:  # too many actions, end the game
                break
            repeat_num = check_action_repeat(action_list=self.moved_actions)
            if repeat_num > 5:
                print("Stop the game because of repeated action.s", file=self.log_file, flush=True)
                break
            if not if_success:
                print("Stop the game because of failed planning.", file=self.log_file, flush=True)
                break

        del self.TreeEnv

        # return scores / self.max_scores
