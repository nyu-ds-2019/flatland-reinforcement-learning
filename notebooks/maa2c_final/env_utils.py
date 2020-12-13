import random
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


def make_env(env_params, random_seed=True):
    """
    Make env, setup calculate constants
    @param env_params: setup parameters
    @param random_seed: whether to use random seed
    @return: env, state_size, action_size, max_steps for given env
    """
    # Obs builder
    if env_params.use_predictor:
        tree_observation = TreeObsForRailEnv(
            max_depth=env_params.observation_tree_depth,
            predictor=ShortestPathPredictorForRailEnv(30)
        )
    else:
        tree_observation = TreeObsForRailEnv(
            max_depth=env_params.observation_tree_depth
        )

    seed = env_params.seed if env_params.seed != -1 else random.randint(0, 100)

    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim

    # for now
    # n_agents = env_params.n_agents_min

    print(f"Env generation seed: {seed}, agents: {n_agents}, rows: {y_dim}, cols: {x_dim}")

    if env_params.rail_generator == "sparse":
        # n_cities = random.randint(env_params.n_cities_min, env_params.n_cities_max)

        rail_gen = sparse_rail_generator(
            max_num_cities=env_params.n_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=env_params.max_rails_between_cities,
            max_rails_in_city=env_params.max_rails_in_city
        )
    else:
        n_goals = n_agents + random.randint(0, 3)
        min_dist = int(0.75 * min(x_dim, y_dim))

        rail_gen = complex_rail_generator(
            nr_start_goal=n_goals,
            nr_extra=25,
            min_dist=min_dist,
            max_dist=9999,
            seed=seed
        )

    # setup env
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=rail_gen,
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation
    )

    env.reset(regenerate_rail=True, regenerate_schedule=True)

    # calc state size given the depth of the tree and num features
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = 0
    for i in range(env_params.observation_tree_depth + 1):
        n_nodes += np.power(4, i)

    frame_stack_mult = 1
    if env_params.stack_obs:
        frame_stack_mult = env_params.how_many_stack
    state_size = frame_stack_mult * n_features_per_node * n_nodes

    # there are always 5 actions
    action_size = 5
    # official formula
    if env_params.rail_generator == "sparse":
        max_steps = int(4 * 2 * (env.height + env.width + (env_params.n_agents / env_params.n_cities)))
    else:
        max_steps = int(3 * (env.height + env.width))

    random.seed()
    return env, state_size, action_size, max_steps


def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node, current_tree_depth: int, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes * 4

    data, distance, agent_data = _split_node_into_feature_groups(node)

    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree, max_tree_depth: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function splits the tree into three difference arrays of values
    """
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_observation(observation, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalize_to_range=True)
    agent_data = np.clip(agent_data, -1, 1)
    normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalized_obs
