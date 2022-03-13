import datetime
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
print(sys.path)

import textworld
import yaml
from evaluate.evaluate_rl import evaluate_rl_with_supervised_graphs, evaluate_rl_with_unsupervised_graphs
from agent.agent import OORLAgent
from generic import reinforcement_learning_dataset
from generic.data_utils import load_config, generate_triplets_filter_mask, read_args
from generic.model_utils import load_graph_extractor
from planner.mcts_planner import MCTSPlanning


def test_mcts(args):
    now = datetime.datetime.now()
    time_store = str(now.year) + '-' + str(now.month) + '-' \
                 + str(now.day) + '-' + str(now.hour) + ':' + str(now.minute)

    planner_config, debug_mode, log_file_path, _ = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''  # for local machine debugging
    rule_based_extraction = False
    print("Start Testing with debug mode {0}.".format('on' if debug_mode else 'off'), file=log_file, flush=True)
    planner_config['general']['use_cuda'] = False
    if debug_mode:
        debug_msg = '_debug'
        # planner_config['general']['training']['batch_size'] = 3
        # planner_config['general']['checkpoint']['report_frequency'] = 1
        # eval_max_counter = 10
        # planner_config['general']['training']['sample_number'] = 1

    if planner_config['rl']['difficulty_level'] == 'mixed':
        candidate_triplets = []
        for difficulty_level in [3, 5, 7, 9]:
            set_dir = planner_config["rl"]["triplet_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
                difficulty_level)
            with open(set_dir, 'r') as f:
                candidate_triplets_sub = f.readlines()
            candidate_triplets += candidate_triplets_sub
    else:
        set_dir = planner_config["rl"]["triplet_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
            planner_config['rl']['difficulty_level'])
        with open(set_dir, 'r') as f:
            candidate_triplets = f.readlines()

    agent_planner = OORLAgent(planner_config,
                              log_file=log_file,
                              debug_msg=debug_msg,
                              init_optimizer=False)

    planning_load_path = agent_planner.output_dir + agent_planner.experiment_tag + \
                         "/difficulty_level_{0}/saved_model_dqn{1}_{2}.pt".format(
                             agent_planner.difficulty_level,
                             '_unsupervised' if 'unsupervised' in agent_planner.task else '',
                             args.FIX_POINT)

    agent_planner.load_pretrained_model(planning_load_path,
                                        log_file=log_file,
                                        load_partial_graph=False)

    # make game environments
    eval_requested_infos = agent_planner.select_additional_infos_lite()
    eval_requested_infos.extras = ["walkthrough"]
    games_dir = "../source/dataset/"

    test_game_file_names = reinforcement_learning_dataset.get_evaluation_game_env(
        games_dir + planner_config['rl']['data_path'],
        planner_config['rl']['difficulty_level'],
        'test',
        debug_mode=debug_mode)

    if not debug_mode:
        # planning_tree_log = open('./planning_logs/planning_trees_df{0}_mcts_cpuct{1}_simul{2}_maxlen{3}_discount{4}_{5}.txt'.
        #                          format(agent_planner.difficulty_level,
        #                                 agent_planner.c_puct,
        #                                 agent_planner.simulations_num,
        #                                 agent_planner.max_search_depth,
        #                                 agent_planner.discount_rate,
        #                                 time_store),
        #                          'wt')
        planning_action_log = open(
            './planning_logs/planning{7}_actions_df{0}_mcts_cpuct{1}_simul{2}_maxlen{3}_discount{4}_rand{5}{8}_{6}.txt'.
                format(agent_planner.difficulty_level,
                       agent_planner.c_puct,
                       agent_planner.simulations_num,
                       agent_planner.max_search_depth,
                       agent_planner.discount_rate,
                       agent_planner.random_move_prob_add,
                       time_store,
                       '_unsupervised' if 'unsupervised' in agent_planner.task else '',
                       '_rule' if rule_based_extraction else ''),
            'wt')
        planning_tree_log = None
    else:
        planning_action_log = open(
            './planning_logs/planning_actions_df{0}_tmp'.
                format(agent_planner.difficulty_level),
            'wt')
        planning_tree_log = None

    # gamefiles = ['../source/dataset/rl.0.2/test/difficulty_level_7/tw-cooking-recipe1+take1+cook+cut+open-O7yoUjMDtn8pSQK8IYYr.z8']

    pre_game_difficulty_level = None
    all_normalized_score = []
    for game_idx in range(len(test_game_file_names)):
        gamefile = test_game_file_names[game_idx]
        # if 'O7yoUjMDtn8pSQK8IYYr' not in gamefile:
        #     continue
        # for gamefile in gamefiles:
        #     game_idx = gamefiles.index(gamefile)
        game_difficulty_level = int(gamefile.split('/')[5].split('_')[-1])
        if agent_planner.load_extractor and game_difficulty_level != pre_game_difficulty_level:
            print("\n\n" + "*" * 30 + "Start Loading extractor" + "*" * 30, file=log_file, flush=True)
            with open('../configs/predict_graphs_dynamics_linear_seen_fineTune_df{0}.yaml'.format(
                    game_difficulty_level)) as reader:
                extract_config = yaml.safe_load(reader)
            extract_config['general']['use_cuda'] = False
            extractor = OORLAgent(extract_config, log_file, '', skip_load=True)
            load_graph_extractor(extractor, log_file, game_difficulty_level)
            print("*" * 30 + "Finish Loading extractor" + "*" * 30, file=log_file, flush=True)
            pre_game_difficulty_level = game_difficulty_level
        else:
            extractor = None

        print("\n Running on the {0} game: {1}\n".format(game_idx, gamefile.split('-')[-1].split('.')[0]),
              file=planning_action_log, flush=True)
        # print("\n Running on the {0} game: {1}\n".format(game_idx, gamefile), file=planning_tree_log, flush=True)

        env = textworld.start(gamefile, infos=eval_requested_infos)
        env = textworld.envs.wrappers.Filter(env)
        # if agent_planner.task == 'rl_planning':
        if agent_planner.planner_name == 'MCTS':
            planner = MCTSPlanning(TreeEnv=env,
                                   agent_planner=agent_planner,
                                   extractor=extractor,
                                   candidate_triplets=candidate_triplets,
                                   random_seed=game_idx,
                                   log_file=log_file,
                                   planning_action_log=planning_action_log,
                                   planning_tree_log=planning_tree_log,
                                   rule_based_extraction=rule_based_extraction,
                                   difficulty_level=game_difficulty_level,
                                   debug_mode=debug_mode)

            # action_selected_all = ["open fridge", "examine cookbook", "take red potato from counter", "cook red potato with stove",
            #  "cook red potato with stove"]
            # action_selected_all = ["open fridge", "examine cookbook", "take red potato from counter", "cook red potato with oven",
            #  "cook red potato with stove", "cook red potato with stove", "cook red potato with stove"]
            # planner.get_candidate_actions(action_selected_all=action_selected_all)

            planner.plan()
            # all_normalized_score.append(normalize_scores)
            # print("Finish game {0} with normalized scores {1}".format(gamefile, normalize_scores), flush=True,
            #       file=log_file)
            del env
            # break

    if not debug_mode:
        planning_action_log.close()
        # planning_tree_log.close()
    # break


def test_dqn(args):
    now = datetime.datetime.now()
    time_store = str(now.year) + '-' + str(now.month) + '-' \
                 + str(now.day) + '-' + str(now.hour) + ':' + str(now.minute)
    load_extractor = True
    config, debug_mode, log_file_path, test_model_name = load_config(args)
    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None

    random_rate = 0.3

    debug_msg = ''  # for local machine debugging
    if debug_mode:
        debug_msg = '_debug'
        config['general']['training']['batch_size'] = 16
        config['general']['evaluate']['batch_size'] = 5
        config['rl']['evaluate']['max_nb_steps_per_episode'] = 50

    print("\n\n" + "*" * 30 + "Start Loading Model" + "*" * 30, file=log_file, flush=True)
    # load_model_tag = 'saved_model_dqn_df-9-mem-500000_cstr_Jun-08-2021.pt'
    # load_model_tag = 'saved_model_dqn_df-7-mem-500000-epi-50000-anneal-0.3-_cstr_Jun-20-2021.pt'
    load_model_tag = 'saved_model_dqn_df-3-mem-300000-epi-20000-maxstep-200-anneal-0.3-cstr_scratch_neg_reward-me-0.3-seed-123_Sept-26-2021_best.pt'
    agent = OORLAgent(config, log_file=log_file, debug_msg=debug_msg, skip_load=True)
    load_from_path = agent.output_dir + agent.experiment_tag + "/difficulty_level_{0}/{1}".format(
        agent.difficulty_level,
        test_model_name if test_model_name is not None else load_model_tag
    )
    agent.load_pretrained_model(load_from_path, log_file=log_file)
    print("*" * 30 + "Finish Loading Model" + "*" * 30, file=log_file, flush=True)

    set_dir = config["rl"]["triplet_path"] + "/difficulty_level_{0}/all_poss_set.txt".format(
        config['rl']['difficulty_level'])
    with open(set_dir, 'r') as f:
        candidate_triplets = f.readlines()
    filter_mask = generate_triplets_filter_mask(triplet_set=candidate_triplets,
                                                node2id=agent.node2id,
                                                relation2id=agent.relation2id)

    if load_extractor and 'unsupervised' not in agent.task:
        print("\n\n" + "*" * 30 + "Start Loading extractor" + "*" * 30, file=log_file, flush=True)
        with open('../configs/predict_graphs_dynamics_linear_seen_fineTune_df{0}.yaml'.format(
                agent.difficulty_level)) as reader:
            extract_config = yaml.safe_load(reader)
        extractor = OORLAgent(extract_config, log_file, '')
        load_graph_extractor(extractor, log_file)
        print("*" * 30 + "Finish Loading extractor" + "*" * 30, file=log_file, flush=True)
    else:
        extractor = None

    # make game environments
    requested_infos = agent.select_additional_infos_lite()
    requested_infos_eval = agent.select_additional_infos()
    games_dir = "../source/dataset/"

    eval_env, num_eval_game = reinforcement_learning_dataset.get_evaluation_game_env(
        games_dir + config['rl']['data_path'],
        config['rl']['difficulty_level'],
        "test",
        requested_infos_eval,
        agent.eval_max_nb_steps_per_episode,
        agent.eval_batch_size)

    # planning_action_log = open(
    #     './planning_logs/difficulty_level_{0}/planning_actions{4}_df{0}_dqn_rand-{1}_{2}_{3}.txt'.
    #         format(agent.difficulty_level,
    #                random_rate,
    #                load_model_tag.split('_')[3],
    #                time_store,
    #                '_unsupervised' if 'unsupervised' in agent.task else ''), 'wt')
    # print('Loading dqn model from {0}\n\n'.format(load_model_tag), file=planning_action_log, flush=True)

    if 'unsupervised' not in agent.task:
        eval_game_points, eval_game_points_normalized, eval_game_step, detailed_scores = \
            evaluate_rl_with_unsupervised_graphs(env=eval_env,
                                                 agent=agent,
                                                 num_games=num_eval_game,
                                                 debug_mode=debug_mode,
                                                 random_rate=0,
                                                 log_file=None,
                                                 write_result=True
                                                 )
    else:
        eval_game_points, eval_game_points_normalized, eval_game_step, detailed_scores = \
            evaluate_rl_with_supervised_graphs(eval_env,
                                               agent,
                                               num_eval_game,
                                               extractor,
                                               filter_mask,
                                               debug_mode,
                                               # random_rate=float(agent.epsilon_anneal_to),
                                               log_file=None,
                                               load_extractor=True,
                                               write_result=True,
                                               random_rate=0)

    # planning_action_log.close()


if __name__ == '__main__':
    args = read_args()
    if args.test_mode == 'dqn':
        test_dqn(args)
    elif args.test_mode == 'mcts':
        test_mcts(args)
    else:
        raise ValueError("Unknown testing mode {0}".format(args.test_mode))
