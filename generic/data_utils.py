import argparse
import codecs
import copy
import json
import os
from functools import lru_cache

import numpy as np
import torch
import yaml
from textworld.logic import State, Rule, Proposition, Variable
missing_words = set()


##############################
# KG stuff
##############################
# relations
two_args_relations = ["in", "on", "at", "west_of", "east_of", "north_of", "south_of", "part_of", "needs"]
one_arg_state_relations = ["chopped", "roasted", "diced", "burned", "open", "fried", "grilled", "consumed", "closed",
                           "sliced", "uncut", "raw"]
ignore_relations = ["cuttable", "edible", "drinkable", "sharp", "inedible", "cut", "cooked", "cookable",
                    "needs_cooking"]
opposite_relations = {"west_of": "east_of",
                      "east_of": "west_of",
                      "south_of": "north_of",
                      "north_of": "south_of"}
equivalent_entities = {"inventory": "player",
                       "recipe": "cookbook"}
FOOD_FACTS = ["sliced", "diced", "chopped", "cut", "uncut", "cooked", "burned",
              "grilled", "fried", "roasted", "raw", "edible", "inedible"]


@lru_cache()
def _rules_predicates_inv():
    rules = [
        Rule.parse("query :: in(o, I) -> in(o, I)"),
    ]
    rules += [Rule.parse("query :: in(f, I) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_predicates_recipe():
    rules = [
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) -> part_of(f, RECIPE)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & roasted(ingredient) -> needs_roasted(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & grilled(ingredient) -> needs_grilled(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & fried(ingredient) -> needs_fried(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & sliced(ingredient) -> needs_sliced(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & chopped(ingredient) -> needs_chopped(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & diced(ingredient) -> needs_diced(f)"),
    ]
    return rules


@lru_cache()
def _rules_predicates_scope():
    rules = [
        Rule.parse("query :: at(P, r) -> at(P, r)"),
        Rule.parse("query :: at(P, r) & at(o, r) -> at(o, r)"),
        Rule.parse("query :: at(P, r) & at(d, r) -> at(d, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) -> at(s, r)"),
        Rule.parse("query :: at(P, r) & at(c, r) -> at(c, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) & on(o, s) -> on(o, s)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) -> open(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & closed(c) -> closed(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(o, c) -> in(o, c)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & open(d) -> open(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & closed(d) -> closed(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    rules += [Rule.parse("query :: at(P, r) & at(f, r) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in
              FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(s, r) & on(f, s) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact
              in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(f, c) & {fact}(f) -> {fact}(f)".format(fact=fact))
              for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_exits():
    rules = [
        Rule.parse("query :: at(P, r) & north_of(r', r) -> north_of(r', r)"),
        Rule.parse("query :: at(P, r) & west_of(r', r) -> west_of(r', r)"),
        Rule.parse("query :: at(P, r) & south_of(r', r) -> south_of(r', r)"),
        Rule.parse("query :: at(P, r) & east_of(r', r) -> east_of(r', r)"),
    ]
    return rules


@lru_cache()
def _rules_to_convert_link_predicates():
    rules = [
        Rule.parse("query :: link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    return rules


def process_facts(prev_facts, info_game, info_facts, info_last_action, cmd):
    kb = info_game.kb
    if prev_facts is None or cmd == "restart":
        facts = set()
    else:
        if cmd == "inventory":  # Bypassing TextWorld's action detection.
            facts = set(find_predicates_in_inventory(State(kb.logic, info_facts)))
            return prev_facts | facts

        elif info_last_action is None:
            return prev_facts  # Invalid action, nothing has changed.

        elif info_last_action.name == "examine" and "cookbook" in [v.name for v in info_last_action.variables]:
            facts = set(find_predicates_in_recipe(State(kb.logic, info_facts)))
            return prev_facts | facts

        state = State(kb.logic, prev_facts | set(info_last_action.preconditions))
        success = state.apply(info_last_action)
        assert success
        facts = set(state.facts)

    # Always add facts in sight.
    facts |= set(find_predicates_in_scope(State(kb.logic, info_facts)))
    facts |= set(find_exits_in_scope(State(kb.logic, info_facts)))

    return facts


def find_exits_in_scope(state):
    actions = state.all_applicable_actions(_rules_exits())

    def _convert_to_exit_fact(proposition):
        return Proposition(proposition.name,
                           [Variable("exit", "LOCATION"),
                            proposition.arguments[1],
                            proposition.arguments[0]])

    return [_convert_to_exit_fact(action.postconditions[0]) for action in actions]


def find_predicates_in_inventory(state):
    actions = state.all_applicable_actions(_rules_predicates_inv())
    return [action.postconditions[0] for action in actions]


def find_predicates_in_recipe(state):
    actions = state.all_applicable_actions(_rules_predicates_recipe())

    def _convert_to_needs_relation(proposition):
        if not proposition.name.startswith("needs_"):
            return proposition

        return Proposition("needs",
                           [proposition.arguments[0],
                            Variable(proposition.name.split("needs_")[-1], "STATE")])

    return [_convert_to_needs_relation(action.postconditions[0]) for action in actions]


def find_predicates_in_scope(state):
    actions = state.all_applicable_actions(_rules_predicates_scope())
    return [action.postconditions[0] for action in actions]


def serialize_facts(facts):
    PREDICATES_TO_DISCARD = {"ingredient_1", "ingredient_2", "ingredient_3", "ingredient_4", "ingredient_5",
                             "out", "free", "used", "cooking_location", "link"}
    CONSTANT_NAMES = {"P": "player", "I": "player", "ingredient": None, "slot": None, "RECIPE": "cookbook"}
    # e.g. [("wooden door", "backyard", "in"), ...]
    serialized = [[arg.name if arg.name and arg.type not in CONSTANT_NAMES else CONSTANT_NAMES[arg.type] for arg in
                   fact.arguments] + [fact.name]
                  for fact in sorted(facts) if fact.name not in PREDICATES_TO_DISCARD]
    return filter_triplets([fact for fact in serialized if None not in fact])


def filter_triplets(triplets):
    tp = []
    for item in triplets:
        # item = process_equivalent_entities_in_triplet(item)
        item = process_exits_in_triplet(item)
        if item[-1] in (two_args_relations + one_arg_state_relations):
            tp.append([it.lower() for it in item])
        else:
            if item[-1] not in ignore_relations:
                print("Warning..., %s not in known relations..." % (item[-1]))

    for i in range(len(tp)):
        if tp[i][-1] in one_arg_state_relations:
            tp[i].append("is")

    tp = process_burning_triplets(tp)
    # tp = process_direction_triplets(tp)
    return tp


def process_exits_in_triplet(triplet):
    # ["exit", "kitchen", "backyard", "south_of"]
    if triplet[0] == "exit":
        return [triplet[0], triplet[1], triplet[3]]
    else:
        return triplet


def process_burning_triplets(list_of_triplets):
    burned_stuff = []
    for t in list_of_triplets:
        if "burned" in t:
            burned_stuff.append(t[0])
    res = []
    for t in list_of_triplets:
        if t[0] in burned_stuff and t[1] in ["grilled", "fried", "roasted"]:
            continue
        res.append(t)
    return res


def adj_to_triplets(adj_matrix, node_vocab, relation_vocab):
    """
    adj_matrix: [batch_size, relation_id, node_id, node_id]
    """
    triplets = []
    for bid in range(len(adj_matrix)):
        triplets.append([])
        result = np.where(adj_matrix[bid] == 1)
        for i in range(len(result[0])):
            relation = relation_vocab[result[0][i]]
            sub_node = node_vocab[result[1][i]]
            obj_node = node_vocab[result[2][i]]
            triplets[bid].append([sub_node, obj_node, relation])
    return triplets


def matching_object_from_obs(observations, actions,
                             node_vocab, input_triplets,
                             pred_triplets, diff_level=1):
    batch_size = len(input_triplets)
    match_triplets_all = []
    object_tense_dict = object_tense_transfer_dict()
    for bid in range(batch_size):
        match_set = set()
        observation = observations[bid]

        for word in node_vocab:
            word = word.strip()
            if word in object_tense_dict.keys():
                word_tense = object_tense_dict[word]
            else:
                word_tense = 'N/A'
            if word in observation or word_tense in observation:
                match_set.add(word)

        match_triplets = []
        if actions[bid] == 'open fridge' or actions[bid] == 'examine cookbook':
            check_both_flag = True
        elif len(input_triplets[bid]) == 0:  # in the beginning of a game
            check_both_flag = True
        elif diff_level > 1:
            check_both_flag = True
        else:
            check_both_flag = False
        for pred_triplet in pred_triplets[bid]:
            find_flag = False
            for input_triplet in input_triplets[bid]:
                if input_triplet[0] == pred_triplet[0] and input_triplet[1] == pred_triplet[1] \
                        and input_triplet[2] == pred_triplet[2]:
                    match_triplets.append(pred_triplet)
                    find_flag = True
                    break
            if not find_flag:
                if check_both_flag:
                    if pred_triplet[0] == 'player' or pred_triplet[1] == 'player' or pred_triplet[1] == 'cookbook' \
                            or pred_triplet[1] == 'uncut' or pred_triplet[1] == 'raw' or pred_triplet[1] == 'closed':
                        if pred_triplet[0] in match_set or pred_triplet[1] in match_set:
                            match_triplets.append(pred_triplet)
                    elif pred_triplet[0] in match_set and pred_triplet[1] in match_set:
                        match_triplets.append(pred_triplet)
                else:
                    if pred_triplet[0] in match_set or pred_triplet[1] in match_set:
                        match_triplets.append(pred_triplet)
        match_triplets_all.append(match_triplets)

    return match_triplets_all


def diff_triplets(triplets1, triplets2):
    """
    compute the difference between two groups of triplets
    """
    triplets_set_1 = set(['$'.join(triplet) for triplet in triplets1])
    triplets_set_2 = set(['$'.join(triplet) for triplet in triplets2])

    diff_triplet_list1 = list(triplets_set_1 - triplets_set_2)
    diff_triplet_list2 = list(triplets_set_2 - triplets_set_1)

    return [triplet_str.split('$') for triplet_str in diff_triplet_list1 + diff_triplet_list2], \
           len(diff_triplet_list1), len(diff_triplet_list2)


def check_node_exits(nodeid, relationid, adj, node_pos, node_num=99):
    found_ids = []
    if node_pos == 1:
        for node2checkid in range(node_num):
            if adj[0, relationid, node2checkid, nodeid]:
                found_ids.append(node2checkid)
    elif node_pos == 2:
        for node2checkid in range(node_num):
            if adj[0, relationid, nodeid, node2checkid]:
                found_ids.append(node2checkid)
    return found_ids


def object_tense_transfer_dict():
    tense_map = {
        'chopped': 'chop',
        'roasted': 'roast',
        'diced': 'dice',
        'burned': 'burn',
        'open': 'open',
        'fried': 'fry',
        'grilled': 'grill',
        'consumed': 'consume',
        'closed': 'close',
        'sliced': 'slice',
        'frosted-glass door': 'frosted - glass door'
    }
    return tense_map


def load_all_possible_set(set_dir, log_file=None):
    """load the all candidate triplets"""
    all_set_data_dict = {}
    try:
        with open(set_dir, 'r') as f:
            set_data = f.readlines()
        print("Candidate triplet set is loaded from {0}".format(set_dir), file=log_file, flush=True)
    except:
        print("No candidate triplet set is loaded", file=log_file, flush=True)
        return all_set_data_dict

    for line in set_data:
        items = line.split('$')
        if items[0].strip() in all_set_data_dict.keys():
            all_set_data_dict[items[0].strip()].add(line.replace('\n', ''))
        else:
            all_set_data_dict.update({items[0].strip(): set()})
            all_set_data_dict[items[0].strip()].add(line.replace('\n', ''))

        if items[1].strip() in all_set_data_dict.keys():
            all_set_data_dict[items[1].strip()].add(line.replace('\n', ''))
        else:
            all_set_data_dict.update({items[1].strip(): set()})
            all_set_data_dict[items[1].strip()].add(line.replace('\n', ''))

    return all_set_data_dict


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument('--test_mode', choices=['dqn', 'mcts'], help='the model to be test.')
    parser.add_argument("-d", "--debug_mode", help="if debugging", dest="BEBUG_MODE", default=False, required=False)
    parser.add_argument("-l", "--log_file", help="log file", dest="LOG_FILE_PATH", default=None, required=False)
    parser.add_argument("-t", "--train_flag", help="if training",
                        dest="TRAIN_FLAG",
                        default='1', required=False)
    parser.add_argument("-f", "--fix_checkpoint", help="if restart from a fixed checkpoint",
                        dest="FIX_POINT",
                        default=None, required=False)
    parser.add_argument("-n", "--test_model_name", help="the name of the model to be tested",
                        dest="TEST_MODEL_NAME",
                        default=None, required=False)
    parser.add_argument("-s", "--seed", help="the seed of randomness",
                        dest="SEED",
                        default=123,
                        required=False,
                        type=int)
    args = parser.parse_args()
    return args


def load_config(args=None):
    if args is None:
        args = read_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    print("log file is {0}".format(args.LOG_FILE_PATH))
    if int(args.TRAIN_FLAG):
        return config, args.BEBUG_MODE, args.LOG_FILE_PATH
    else:
        return config, args.BEBUG_MODE, args.LOG_FILE_PATH, args.TEST_MODEL_NAME


def _word_to_id(word, word2id):
    try:
        return word2id[word]
    except KeyError:
        key = word + "_" + str(len(word2id))
        if key not in missing_words:
            print("Warning... %s is not in vocab, vocab size is %d..." % (word, len(word2id)))
            missing_words.add(key)
            with open("missing_words.txt", 'a+') as outfile:
                outfile.write(key + '\n')
                outfile.flush()
        return 1


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        ids.append(_word_to_id(word, word2id))
    return ids


def max_len(list_of_list):
    if len(list_of_list) == 0:
        return 0
    return max(map(len, list_of_list))


def load_graph_ids():
    # node vocab
    node_vocab = []
    with codecs.open("../source/vocabularies/node_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            node_vocab.append(line.strip().lower())
    node2id = {}
    for i, w in enumerate(node_vocab):
        node2id[w] = i
    # relation vocab
    relation_vocab = []
    with codecs.open("../source/vocabularies/relation_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            relation_vocab.append(line.strip().lower())
    origin_relation_number = len(relation_vocab)
    # add reverse relations
    for i in range(origin_relation_number):
        relation_vocab.append(relation_vocab[i] + "_reverse")
    relation2id = {}
    for i, w in enumerate(relation_vocab):
        relation2id[w] = i

    return node_vocab, node2id, relation_vocab, relation2id, origin_relation_number


class GraphDataset:
    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.relation_types = {}
        self.graphs = {}

    def _get_id(self, k, D):
        if k not in D:
            D[k] = len(D)

        return D[k]

    def _get_entity_id(self, entity):
        return self._get_id(entity, self.entities)

    def _get_relation_id(self, relation):
        return self._get_id(relation, self.relations)

    def _get_relation_type_id(self, relation_type):
        return self._get_id(relation_type, self.relation_types)

    def _get_graph_id(self, graph):
        return self._get_id(graph, self.graphs)

    def dumps(self):
        meta = {}
        meta["graphs"] = {v: list(k) for k, v in self.graphs.items()}
        meta["entities"] = {v: k for k, v in self.entities.items()}
        meta["relations"] = {v: k for k, v in self.relations.items()}
        meta["relation_types"] = {v: k for k, v in self.relation_types.items()}
        return json.dumps(meta)

    def dump(self, filename):
        with open(filename, 'w') as f:
            f.write(self.dumps())

    @classmethod
    def loads(cls, data):
        data = json.loads(data)
        self = cls()
        self.entities = data["entities"]
        self.relations = data["relations"]
        self.relation_types = data["relation_types"]
        self.graphs = data["graphs"]
        return self

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            return cls.loads(f.read())

    def _get_link(self, idx):
        e1, e2, r = self.relations[str(idx)]
        return self.entities[str(e1)], self.entities[str(e2)], self.relation_types[str(r)]

    def compress(self, G):
        # Assuming G list a list of string triples (e1, e2, r).
        new_G = frozenset(
            self._get_relation_id(
                (self._get_entity_id(e1),
                 self._get_entity_id(e2),
                 self._get_relation_type_id(r)))
            for e1, e2, r in G)
        return self._get_graph_id(new_G)

    def decompress(self, idx):
        return [list(self._get_link(link)) for link in self.graphs[str(idx)]]


def merge_sample_triplet_index(sample_triplet_index_list):
    """
    merge the samples triplets index for training
    """
    sample_triplet_index_set_agg = []
    curr_batch_size = len(sample_triplet_index_list[0])
    for bid in range(curr_batch_size):
        sample_triplet_index_set_agg.append(set())
    for sample_triplet_index in sample_triplet_index_list:
        for bid in range(curr_batch_size):
            for triplet in sample_triplet_index[bid]:
                triplet_str = list(map(str, triplet))
                sample_triplet_index_set_agg[bid].add('$'.join(triplet_str))
    sample_triplet_index_agg = []
    for bid in range(curr_batch_size):
        sample_triplet_index_agg.append([])
        for triplet_str in list(sample_triplet_index_set_agg[bid]):
            triplet = list(map(int, triplet_str.split('$')))
            sample_triplet_index_agg[bid].append(triplet)
    return sample_triplet_index_agg


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    if isinstance(sequences, np.ndarray):
        return sequences
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def generate_triplets_filter_mask(triplet_set, node2id, relation2id):
    """
    filter the impossible triplets
    """
    mask = np.zeros([1, len(relation2id), len(node2id), len(node2id)])

    for triplet_str in triplet_set:
        triplet = triplet_str.replace('\n', '').split('$')
        relation_id = _word_to_id(triplet[2], relation2id)
        sub_node_id = _word_to_id(triplet[0], node2id)
        obj_node_id = _word_to_id(triplet[1], node2id)
        mask[0, relation_id, sub_node_id, obj_node_id] = 1

    return mask


def rule_based_extraction_from_obs(general_goal_sentence, difficulty_level):
    ingredients = set(general_goal_sentence.split(':')[1].replace('\n\nDirections', '').strip().split("\n"))
    ingredients = [ingredient.strip() for ingredient in ingredients]
    goal_sentences = general_goal_sentence.split('\n\nDirections:\n')
    goal_sentence_meal = goal_sentences[1].strip()
    goal_sentence_ingredient_list = []
    for ingredient in ingredients:
        goal_sentence_ingredient_list.append("ingredients : {0}".format(ingredient))
    goal_sentence_meal_list = []
    for action_sentence in goal_sentence_meal.split('\n'):
        if len(action_sentence.split('the')) == 2:
            operation = action_sentence.split('the')[0].strip()
            ingredient = action_sentence.split('the')[1].strip()
            if difficulty_level == 3:  # fix a bug in the game engine, cook book at diff 3 should not include there operations according to the game rule
                if operation in ['fry', 'roast', 'grill']:
                    continue
            goal_sentence_meal_list.append('{0} the {1}'.format(operation, ingredient))

    return goal_sentence_ingredient_list, goal_sentence_meal_list, ingredients


def extract_goal_sentence_from_obs(agent, object_encodings=None, object_mask=None, pre_goal_sentence=None, obs=None,
                                   obs_origin=None, ingredients=None, goal_sentence_store=None, difficulty_level=1,
                                   rule_based_extraction=False):
    """
    Extract goals from object encodings and observations. This function is for unsupervised training.
    The memory graph (adjacency matrix) is not directly interpretable and thus an extra goal extractor is
    need to be learn and applied here.
    """
    cookbook_flag = False
    if obs is not None:
        if 'ingredients :' in obs and 'directions :' in obs and len(goal_sentence_store) == 0:  # we find the cookbook
            cookbook_flag = True
        else:
            cookbook_flag = False

    if pre_goal_sentence is None:  # it is the initialized goal
        # if difficulty_level == 1 or difficulty_level == 3 or difficulty_level == 7:
        return ['check the cookbook in the kitchen for the recipe'], set(), []
    elif len(pre_goal_sentence) == 0: # we don't capture any meaningful goals, and the game will not be solved :(
        return ['prepare meal'], set(), []
        # elif difficulty_level == 9 or difficulty_level == 5:
        #     return ['find kitchen'], set(), []
    # elif pre_goal_sentence[0] == 'find kitchen' and '-=kitchen=-' in obs.replace(' ', ''):
    #     return ['check the cookbook in the kitchen for the recipe'], set(), []
    elif 'check the cookbook' in pre_goal_sentence[0] and cookbook_flag:  # check cook book
        if 'ingredients :' in obs:
            if rule_based_extraction:
                general_goal_sentence = obs_origin.split('.')[-1]
                goal_sentence_ingredient_list, goal_sentence_meal_list, ingredients = \
                    rule_based_extraction_from_obs(general_goal_sentence, difficulty_level)
            else:
                with torch.no_grad():
                    goal_gen_pred_greedy_words = agent.goal_greedy_generation(observation_strings=[obs],
                                                                              predicted_encodings=object_encodings,
                                                                              node_mask=object_mask,
                                                                              gen_goal_sentence_len=80 if difficulty_level == 9 else 20)
                extracted_goal_sentences = goal_gen_pred_greedy_words[0].replace('<eos>', '').split('*')

                if '<eos>' not in goal_gen_pred_greedy_words[0]:  # extraction may be incomplete
                    extracted_goal_sentences = extracted_goal_sentences[:-1]
                goal_sentence_ingredient_list = []
                goal_sentence_meal_list = []
                ingredients = set()
                for extracted_goal_sentence in extracted_goal_sentences:
                    if 'ingredients' in extracted_goal_sentence:
                        if len(extracted_goal_sentence.split(":")) != 2: # the extraction is incomplete
                            continue
                        else:
                            goal_sentence_ingredient_list.append(extracted_goal_sentence.strip())
                            ingredients.add(extracted_goal_sentence.split(":")[1].strip())
                    else:
                        goal_sentence_meal_list.append(extracted_goal_sentence.strip())
            for extracted_goal_sentence in copy.copy(goal_sentence_meal_list):
                if difficulty_level == 3:  # fix a bug in the game engine, cook book at diff 3 should not include the cook operations according to the game rule
                    if extracted_goal_sentence.split('the')[0].strip() in ['fry', 'roast', 'grill']:
                        goal_sentence_meal_list.remove(extracted_goal_sentence)
                elif difficulty_level == 5:  # fix a bug in the game engine, cook book at diff 5 should not include any operations according to the game rule
                    if extracted_goal_sentence.split('the')[0].strip() in ['fry', 'roast', 'grill', 'dice', 'chop',
                                                                           'slice']:
                        goal_sentence_meal_list.remove(extracted_goal_sentence)
            if difficulty_level == 1 or difficulty_level == 5:
                goal_sentence_meal_list.append('prepare meal')
                goal_sentence_meal = ' '.join(goal_sentence_meal_list)
                goal_sentence_store = [[goal_sentence_meal]]
            else:
                goal_sentence_store = [goal_sentence_meal_list, 'prepare meal']
            if len(goal_sentence_ingredient_list) > 0:
                goal_sentence_ingredients = goal_sentence_ingredient_list
                return goal_sentence_ingredients, ingredients, goal_sentence_store
            else:  # we don't capture any meaningful goals, and the game will not be solved :(
                return ['prepare meal'], set(), []
        else:
            return pre_goal_sentence, ingredients, goal_sentence_store
    elif len(ingredients) > 0:  # collect ingredients
        goal_sentence_ingredient_list = []
        ingredients_copy = copy.copy(ingredients)
        for ingredient in ingredients:
            if ingredient in obs and 'take' in obs and 'take note' not in obs:
                ingredients_copy.remove(ingredient)
            elif ingredient in obs and 'pick up' in obs and difficulty_level == 5:
                ingredients_copy.remove(ingredient)
            else:
                goal_sentence_ingredient_list.append('ingredients : {0}'.format(ingredient))

        pre_goal_sentence = goal_sentence_ingredient_list
        if len(ingredients_copy) == 0:
            if difficulty_level == 1:
                goal_sentence_meal = goal_sentence_store[0]
                return goal_sentence_meal, set(), goal_sentence_store
            else:
                goal_sentence_operation = goal_sentence_store[0]
                return goal_sentence_operation, set(), goal_sentence_store
        else:
            return pre_goal_sentence, ingredients_copy, goal_sentence_store
    elif (difficulty_level == 9 or difficulty_level == 7 or difficulty_level == 3) and len(
            goal_sentence_store) > 0:  # check operations
        operations = pre_goal_sentence
        operations_left = copy.copy(operations)
        finished_operation_num = 0
        object_tense_dict = object_tense_transfer_dict()
        object_tense_reverse_dict = {v: k for k, v in object_tense_dict.items()}
        for operation in operations:
            items = operation.split('the')
            if len(items) == 2:
                op_v = items[0].strip()
                op_n = items[1].strip()
                if (op_v in obs or object_tense_reverse_dict[op_v] in obs) and op_n in obs and not cookbook_flag:
                    finished_operation_num += 1
                    operations_left.remove(operation)
                    break
            else:
                print(operation)

        if finished_operation_num < len(operations) and len(operations_left) > 0:
            pre_goal_sentence = operations_left
            return pre_goal_sentence, ingredients, goal_sentence_store
        else:
            return ['prepare meal'], ingredients, []
    elif "prepare meal" in pre_goal_sentence[0]:  # prepare meal
        if 'adding' in obs and 'meal' in obs:
            return ['eat the meal'], ingredients, goal_sentence_store
        else:
            return pre_goal_sentence, ingredients, goal_sentence_store
    elif 'eat the meal' in pre_goal_sentence[0]:
        if 'eat' in obs and 'meal' in obs:
            return ['end the game'], ingredients, goal_sentence_store
        else:
            return pre_goal_sentence, ingredients, goal_sentence_store
    else:
        return pre_goal_sentence, ingredients, goal_sentence_store


def get_goal_sentence(pre_goal_sentence=None, obs=None, state=None, ingredients=None,
                      node2id=None, relation2id=None, node_vocab=None, relation_vocab=None,
                      goal_sentence_store=None, difficulty_level=1, use_obs_flag=False,
                      recheck_ingredients=True):
    """
    Extract goal from the memory graph. The memory graph must be interpretable.
    """
    object_tense_dict = object_tense_transfer_dict()

    if pre_goal_sentence is not None:
        if len(pre_goal_sentence) == 0:  # the goal is not correctly detected, end this game
            return [], set(), []  # return dumb msg
    if obs is not None and 'you lost !' in obs:  # lost!, end the game
        return ['END'], set(), []   # return ending msg

    cookbook_flag = True
    if use_obs_flag:
        if 'ingredients :' in obs and 'directions :' in obs and len(goal_sentence_store) == 0:  # we find the cookbook
            cookbook_flag = True
        else:
            cookbook_flag = False

    if recheck_ingredients and pre_goal_sentence is not None:
        ingredient_ids = check_node_exits(_word_to_id('cookbook', node2id),
                                          _word_to_id('part_of', relation2id),
                                          state,
                                          1)
        ingredients = set()  # recheck ingredients
        for ingredient_id in ingredient_ids:
            ingredient = node_vocab[ingredient_id]
            node1_id = _word_to_id(ingredient, node2id)
            node2_id = _word_to_id('player', node2id)
            relation_id = _word_to_id('in', relation2id)
            if not state[0, relation_id, node1_id, node2_id]:
                ingredients.add(ingredient)

        if state[0, _word_to_id('in', relation2id), _word_to_id('meal', node2id), _word_to_id('player', node2id)]:
            ingredients = set()

    if pre_goal_sentence is None:  # it is the initialized goal
        if difficulty_level == 1 or difficulty_level == 3 or difficulty_level == 7:
            return ['check the cookbook in the kitchen for the recipe'], set(), []
        elif difficulty_level == 9 or difficulty_level == 5:
            return ['find kitchen'], set(), []
    elif pre_goal_sentence[0] == 'find kitchen' and state[
        0, _word_to_id('at', relation2id), _word_to_id('player', node2id), _word_to_id('kitchen', node2id)]:
        return ['check the cookbook in the kitchen for the recipe'], set(), []
    elif 'check the cookbook' in pre_goal_sentence[0] and cookbook_flag:  # check cook book
        ingredient_ids = check_node_exits(_word_to_id('cookbook', node2id),
                                          _word_to_id('part_of', relation2id),
                                          state,
                                          1)
        if len(ingredient_ids) > 0:  # check ingredients
            goal_sentence_ingredient_list = []
            goal_sentence_meal_list = []
            for ingredient_id in ingredient_ids:
                ingredient = node_vocab[ingredient_id]
                ingredients.add(ingredient)
                goal_sentence_ingredient_list.append('ingredients : {0}'.format(ingredient))
                operation_ids = check_node_exits(_word_to_id(ingredient, node2id),
                                                 _word_to_id('needs', relation2id),
                                                 state,
                                                 2)
                for operation_id in operation_ids:
                    operation = node_vocab[operation_id]
                    if operation in object_tense_dict.keys():
                        operation = object_tense_dict[operation]
                    if difficulty_level == 3:  # fix a bug in the game engine
                        if operation in ['fry', 'roast', 'grill']:
                            continue
                    goal_sentence_meal_list.append('{0} the {1}'.format(operation, ingredient))
            if difficulty_level == 1 or difficulty_level == 5:
                goal_sentence_store = ['prepare meal']
            else:
                goal_sentence_store = [goal_sentence_meal_list, 'prepare meal']
            goal_sentence_ingredients = goal_sentence_ingredient_list
            return goal_sentence_ingredients, ingredients, goal_sentence_store

        else:
            return pre_goal_sentence, ingredients, goal_sentence_store
    elif len(ingredients) > 0:  # collect ingredients
        goal_sentence_ingredient_list = []
        for ingredient in ingredients:
            goal_sentence_ingredient_list.append('ingredients : {0}'.format(ingredient))

        pre_goal_sentence = goal_sentence_ingredient_list
        if not recheck_ingredients:
            ingredients_copy = copy.copy(ingredients)
            for ingredient in ingredients:
                node1_id = _word_to_id(ingredient, node2id)
                node2_id = _word_to_id('player', node2id)
                relation_id = _word_to_id('in', relation2id)
                if state[0, relation_id, node1_id, node2_id]:
                    ingredients_copy.remove(ingredient)
                    for pre_goal_sentence_sub in pre_goal_sentence:
                        if 'ingredients : {0}'.format(ingredient) == pre_goal_sentence_sub.strip():
                            pre_goal_sentence.remove(pre_goal_sentence_sub)
        else:
            ingredients_copy = ingredients

        if len(ingredients_copy) == 0:
            if difficulty_level == 1 or difficulty_level == 5:
                goal_sentence_meal = [goal_sentence_store[0]]
                return goal_sentence_meal, set(), []
            else:
                goal_sentence_operation = goal_sentence_store[0]
                return goal_sentence_operation, set(), goal_sentence_store
        else:
            if recheck_ingredients:
                return pre_goal_sentence, ingredients, goal_sentence_store
            else:
                return pre_goal_sentence, ingredients_copy, goal_sentence_store
    elif (difficulty_level == 9 or difficulty_level == 7 or difficulty_level == 3) and len(
            goal_sentence_store) > 0:  # check operations
        operations = goal_sentence_store[0]
        operations_left = copy.copy(operations)
        finished_operation_num = 0

        for operation in operations:
            items = operation.split('the')
            node1_id = _word_to_id(items[1].strip(), node2id)
            for operation_tense in object_tense_dict.keys():
                if object_tense_dict[operation_tense] == items[0].strip():
                    node2_id = _word_to_id(operation_tense, node2id)
                    break
            relation_id = _word_to_id('is', relation2id)
            if state[0, relation_id, node1_id, node2_id]:
                finished_operation_num += 1
                operations_left.remove(operation)

        if finished_operation_num == len(operations):
            return ['prepare meal'], ingredients, []
        else:
            # pre_goal_sentence = " * ".join(operations_left)
            pre_goal_sentence = operations_left
            return pre_goal_sentence, ingredients, goal_sentence_store
    elif "prepare meal" in pre_goal_sentence[0]:  # prepare meal
        node1_id = _word_to_id('meal', node2id)
        node2_id = _word_to_id('player', node2id)
        relation_id = _word_to_id('in', relation2id)
        if state[0, relation_id, node1_id, node2_id]:
            return ['eat the meal'], ingredients, goal_sentence_store
        else:
            return pre_goal_sentence, ingredients, goal_sentence_store
    elif 'eat the meal' in pre_goal_sentence[0]:
        node1_id = _word_to_id('meal', node2id)
        node2_id = _word_to_id('consumed', node2id)
        relation_id = _word_to_id('is', relation2id)
        if state[0, relation_id, node1_id, node2_id]:
            return ['end the game'], ingredients, goal_sentence_store
        else:
            return pre_goal_sentence, ingredients, goal_sentence_store
    else:
        return pre_goal_sentence, ingredients, goal_sentence_store


def get_game_difficulty_level(game_info):
    """get the game difficult level according to their max score, it is built for the mixed games test"""

    score_difficulty_level_dict = {
        3: 5,
        4: 3,
        5: 7,
        11: 9
    }

    return score_difficulty_level_dict[game_info.max_score]


def preproc(s, tokenizer=None):
    if s is None:
        return "nothing"
    s = s.replace("\n", ' ')
    if "$$$$$$$" in s:
        s = s.split("$$$$$$$")[-1]
    while (True):
        if "  " in s:
            s = s.replace("  ", " ")
        else:
            break
    s = s.strip()
    if len(s) == 0:
        return "nothing"
    s = " ".join([t.text for t in tokenizer(s)])
    s = s.lower()
    return s


def handle_ingame_rewards(next_observations,
                          goal_sentences,
                          ingredients,
                          goal_sentence_stores,
                          actions,
                          log_file,
                          apply_goal_constraint,
                          apply_neg_rewards,
                          selected_actions_histories):
    batch_size = len(next_observations)
    rewards = []
    for bid in range(batch_size):
        next_observation = next_observations[bid]
        goal_sentence = goal_sentences[bid]
        ingredient = ingredients[bid]
        action = actions[bid]
        goal_sentence_store = goal_sentence_stores[bid]
        selected_actions_history = selected_actions_histories[bid][:-1]

        if len(goal_sentence) == 0:
            reward = 0
        elif "you lost !" in next_observation and apply_neg_rewards:
            reward = -1
        elif "score has just gone up by one point" in next_observation:
            if apply_goal_constraint:
                if 'check the cookbook' in goal_sentence[0] or 'find kitchen' in goal_sentence[0]:
                    reward = 0
                else:
                    reward = 1
                if len(ingredient) > 0:  # we should collect gradients, not cook the meal
                    if 'take' not in action.lower():
                        reward = 0
            else:
                reward = 1
        elif '-= kitchen = -' in next_observation and goal_sentence[0] == 'find kitchen':
            reward = 1  # add find kitchen rewards
        elif 'ingredients :' in next_observation and 'directions :' in next_observation and 'check the cookbook' in \
                goal_sentence[0]:
            reward = 1  # add find cookbook rewards
        else:
            reward = 0

        rewards.append(reward)
    return rewards


def handle_rnn_max_len(max_len, batch_elements, padding):
    return_batch_elements = []
    lens = []
    for elem in batch_elements:
        if len(elem) <= max_len:
            return_batch_elements.append(elem + [padding] * (max_len - len(elem)))
            lens.append(len(elem))
        else:
            return_batch_elements.append(elem[len(elem) - max_len:])  # we need the last max_len steps
            lens.append(max_len)
    return return_batch_elements, lens


def compute_mask(x):
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


def to_one_hot(y_true, n_classes):
    # y_true: batch x time
    batch_size, length = y_true.size(0), y_true.size(1)
    y_onehot = torch.FloatTensor(batch_size, length, n_classes)  # batch x time x n_class
    if y_true.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    y_onehot = y_onehot.view(-1, y_onehot.size(-1))  # batch*time x n_class
    y_true = y_true.view(-1, 1)  # batch*time x 1
    y_onehot.scatter_(1, y_true, 1)  # batch*time x n_class
    return y_onehot.view(batch_size, length, -1)


def NegativeLogLoss(y_pred, y_true, mask=None, smoothing_eps=0.0):
    """
    Shape:
        - y_pred:    batch x time x vocab
        - y_true:    batch x time
        - mask:      batch x time
    """
    y_true_onehot = to_one_hot(y_true, y_pred.size(-1))  # batch x time x vocab
    if smoothing_eps > 0.0:
        y_true_onehot = y_true_onehot * (1.0 - smoothing_eps) + (1.0 - y_true_onehot) * smoothing_eps / (
                y_pred.size(-1) - 1)
    P = y_true_onehot * y_pred  # batch x time x vocab
    P = torch.sum(P, dim=-1)  # batch x time
    gt_zero = torch.gt(P, 0.0).float()  # batch x time
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x time
    log_P = torch.log(P + epsilon) * gt_zero  # batch x time
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return output


def check_action_repeat(action_list, repeat_nums=range(1, 5)):
    max_num = 0
    for repeat_num in repeat_nums:
        count_dict = {}
        for i in range(len(action_list)):
            if i % repeat_num != 0:
                continue
            idx = len(action_list) - i
            if idx - repeat_num * 2 < 0:
                break
            string_key_1 = '$'.join(action_list[idx - repeat_num:idx])
            string_key_2 = '$'.join(action_list[idx - repeat_num * 2:idx - repeat_num])
            if string_key_1 == string_key_2:
                if string_key_1 in count_dict.keys():
                    count_dict[string_key_1] += 1
                else:
                    count_dict[string_key_1] = 1
        if len(count_dict.values()) > 0:
            max_repeat_num = max(count_dict.values())
            if max_repeat_num > max_num:
                max_num = max_repeat_num
    return max_num


def process_fully_obs_facts(info_game, facts):
    state = State(info_game.kb.logic, facts)
    state = convert_link_predicates(state)
    inventory_facts = set(find_predicates_in_inventory(state))
    recipe_facts = set(find_predicates_in_recipe(state))
    return set(state.facts) | inventory_facts | recipe_facts


def convert_link_predicates(state):
    actions = state.all_applicable_actions(_rules_to_convert_link_predicates())
    for action in list(actions):
        state.apply(action)
    return state