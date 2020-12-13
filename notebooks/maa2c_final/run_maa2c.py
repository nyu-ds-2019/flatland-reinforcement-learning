import argparse
import gym
import numpy as np
import os
# import tensorflow as tf
import time
import pickle
import json
from argparse import Namespace

from MAA2C import MAA2C
from common.utils import agg_double_list

import sys
# import matplotlib.pyplot as plt

from env_utils import make_env


MAX_EPISODES = 2500
EPISODES_BEFORE_TRAIN = 1
EVAL_EPISODES = 10
EVAL_INTERVAL = 20

# roll out n steps
ROLL_OUT_N_STEPS = 25
# only remember the latest ROLL_OUT_N_STEPS
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
# only use the latest ROLL_OUT_N_STEPS for training A2C
BATCH_SIZE = ROLL_OUT_N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.01
#
DONE_PENALTY = -10.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2018


env_params = {}
with open('env_params.json') as params_file:
    env_params = json.load(params_file)
print(env_params)


def run():
    # Create environment
    env, state_dim, action_dim, max_steps = make_env(env_params=Namespace(**env_params))
    env_eval, state_dim, action_dim, max_steps = make_env(env_params=Namespace(**env_params))
    # Create agent trainers
    # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    # num_adversaries = min(env.n, arglist.num_adversaries)

    obs_shape_n = state_dim
    act_shape_n = action_dim

    maa2c = MAA2C(env, env_params['n_agents'], obs_shape_n, act_shape_n, max_steps = max_steps)

    episodes =[]
    eval_rewards =[]
    while maa2c.n_episodes < MAX_EPISODES:
        # print(maa2c.env_state)
        maa2c.interact()
        # if maa2c.n_episodes >= EPISODES_BEFORE_TRAIN:
        #     maa2c.train()
        maa2c.train()
        if maa2c.episode_done and ((maa2c.n_episodes)%EVAL_INTERVAL == 0):
            rewards, _ = maa2c.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f, STD %.2f" % (maa2c.n_episodes, rewards_mu, rewards_std))
            episodes.append(maa2c.n_episodes)
            eval_rewards.append(rewards_mu)

    

if __name__ == '__main__':
    run()