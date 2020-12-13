# from datetime import datetime
import os
import random
import sys
import copy
import pickle
import datetime
import matplotlib.pyplot as plt

from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint
from collections import namedtuple, deque, Iterable
from itertools import chain

import psutil
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv










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









# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=dim)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y

def firmmax_sample(logits, temperature, dim=1):
    if temperature == 0:
        return F.softmax(logits, dim=dim)
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)) / temperature
    return F.softmax(y, dim=dim)

def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True

def sep_clip_grad_norm(parameters, max_norm, norm_type=2):
    """
    Clips gradient norms calculated on a per-parameter basis, rather than over
    the whole list of parameters as in torch.nn.utils.clip_grad_norm.
    Code based on torch.nn.utils.clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    for p in parameters:
        if norm_type == float('inf'):
            p_norm = p.grad.data.abs().max()
        else:
            p_norm = p.grad.data.norm(norm_type)
        clip_coef = max_norm / (p_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)








class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets








class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]

        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            # head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
            #                    .mean()) for probs in all_attend_probs[i]]
            head_entropies = [(-((probs + 1e-8).log() * probs).sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets








class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])









class Policy:
    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps=0.):
        raise NotImplementedError











class AttentionSACPolicy(Policy):
    def __init__(self, n_agents, state_size, action_size, parameters):
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        sa_sizes = [(state_size, action_size)] * n_agents

        self.hidsize = parameters.hidden_size
        self.buffer_size = parameters.buffer_size
        self.batch_size = parameters.batch_size
        self.update_every = parameters.update_every
        self.learning_rate = parameters.learning_rate
        self.tau = parameters.tau
        self.gamma = parameters.gamma
        self.buffer_min_size = parameters.buffer_min_size
        self.use_gpu = parameters.use_gpu

        self.t_step = 0
        self.pol_dev = 'gpu'
        self.critic_dev = 'gpu'
        self.trgt_pol_dev = 'gpu'
        self.trgt_critic_dev = 'gpu'

        self.q_lr = 0.001
        self.niter = 0
        self.reward_scale = 10.

        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")


        self.memory = MultiAgentReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.device, self.n_agents)

        self.agents = [AttentionAgent(
                           num_in_pol = self.state_size,
                           num_out_pol = self.action_size,
                           hidden_dim = 256,
                           lr = 0.001
                        ) for _ in range(self.n_agents)]
        self.critic = AttentionCritic(
            sa_sizes, 
            hidden_dim = 128, 
            norm_in = True, 
            attend_heads = 8
        )
        self.target_critic = AttentionCritic(
            sa_sizes, 
            hidden_dim = 128, 
            norm_in = True, 
            attend_heads = 8
        )
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr = self.q_lr,
            weight_decay = 1e-3
        )

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def act(self, state, agent_id):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.agents[agent_id].policy.eval()
        action_values = self.agents[agent_id].step(state, explore = True)
        self.agents[agent_id].policy.train()
        return np.argmax(action_values.cpu().data.numpy())


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        # print('in learn')
        self.prep_training('gpu' if self.use_gpu else 'cpu')

        # sample = self.replay_buffer.sample(config.batch_size, to_gpu = config.use_gpu)
        sample = self.memory.sample()

        self.update_critic(sample)
        self.update_policies(sample)
        self.update_all_targets()
        self.prep_rollouts(device='gpu')

    
    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.n_agents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.n_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.n_agents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            # logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
            #                   self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.n_agents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            # if logger is not None:
            #     logger.add_scalar('agent%i/losses/pol_loss' % a_i,
            #                       pol_loss, self.niter)
            #     logger.add_scalar('agent%i/grad_norms/pi' % a_i,
            #                       grad_norm, self.niter)

    def prep_training(self, device='gpu'):
        torch_device = torch.device('cuda:0' if device == 'gpu' else 'cpu')
        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()
        # if not self.pol_dev == device:
        #     for a in self.agents:
        #         a.policy = fn(a.policy)
        #     self.pol_dev = device
        # if not self.critic_dev == device:
        #     self.critic = fn(self.critic)
        #     self.critic_dev = device
        # if not self.trgt_pol_dev == device:
        #     for a in self.agents:
        #         a.target_policy = fn(a.target_policy)
        #     self.trgt_pol_dev = device
        # if not self.trgt_critic_dev == device:
        #     self.target_critic = fn(self.target_critic)
        #     self.trgt_critic_dev = device
        
        for a in self.agents:
            a.policy = a.policy.to(torch_device)
            a.target_policy = a.target_policy.to(torch_device)
        
        self.critic = self.critic.to(torch_device)
        self.target_critic = self.target_critic.to(torch_device)

        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()

        
        # if device == 'gpu':
        #     fn = lambda x: x.cuda()
        # else:
        #     fn = lambda x: x.cpu()
        
        # only need main policy for rollouts
        # if not self.pol_dev == device:
        #     for a in self.agents:
        #         a.policy = fn(a.policy)
        #     self.pol_dev = device

    def update_all_targets(self):
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)












Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .float().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states






Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class MultiAgentReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, n_agents):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.n_agents = n_agents

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = Experience(
            [np.expand_dims(state, 0) for state in states], 
            [np.expand_dims(action, 0) for action in actions], 
            [reward for reward in rewards], 
            [np.expand_dims(next_state, 0) for next_state in next_states], 
            [done for done in dones]
        )
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for agent in range(self.n_agents):
            states.append(torch.from_numpy(self.__v_stack_impr([e.state[agent] for e in experiences if e is not None])) \
                .float().to(self.device))
            actions.append(torch.from_numpy(self.__v_stack_impr([e.action[agent] for e in experiences if e is not None])) \
                .long().to(self.device))
            rewards.append(torch.from_numpy(self.__v_stack_impr([e.reward[agent] for e in experiences if e is not None])) \
                .float().to(self.device))
            next_states.append(torch.from_numpy(self.__v_stack_impr([e.next_state[agent] for e in experiences if e is not None])) \
                .float().to(self.device))
            dones.append(torch.from_numpy(self.__v_stack_impr([e.done[agent] for e in experiences if e is not None]).astype(np.uint8)) \
                .float().to(self.device))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states







from timeit import default_timer


class Timer(object):
    """
    Utility to measure times.

    TODO:
    - add "lap" method to make it easier to measure average time (+std) when measuring the same thing multiple times.
    """

    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        self.start_time = default_timer()

    def end(self):
        self.total_time += default_timer() - self.start_time

    def get(self):
        return self.total_time

    def get_current(self):
        return default_timer() - self.start_time

    def reset(self):
        self.__init__()

    def __repr__(self):
        return self.get()


    








def create_rail_env(env_params, tree_observation):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        # rail_generator=sparse_rail_generator(
        #     max_num_cities=n_cities,
        #     grid_mode=False,
        #     max_rails_between_cities=max_rails_between_cities,
        #     max_rails_in_city=max_rails_in_city
        # ),
        # schedule_generator=sparse_schedule_generator(),
        rail_generator = complex_rail_generator(
            nr_start_goal=10,
            nr_extra=10,
            min_dist=10,
            max_dist=99999,
            seed=1
        ),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )










def train_agent(train_params, train_env_params, eval_env_params, obs_params):
    # Environment parameters
    n_agents = train_env_params.n_agents
    x_dim = train_env_params.x_dim
    y_dim = train_env_params.y_dim
    n_cities = train_env_params.n_cities
    max_rails_between_cities = train_env_params.max_rails_between_cities
    max_rails_in_city = train_env_params.max_rails_in_city
    seed = train_env_params.seed

    # Unique ID for this training
    now = datetime.datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes
    restore_replay_buffer = train_params.restore_replay_buffer
    save_replay_buffer = train_params.save_replay_buffer

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environments
    train_env = create_rail_env(train_env_params, tree_observation)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(eval_env_params, tree_observation)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    # Setup renderer
    if train_params.render:
        env_renderer = RenderTool(train_env, gl="PGL")

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = train_env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
    max_steps = train_env._max_episode_steps

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    # agent_prev_action = [2] * n_agents
    agent_prev_action = [np.array([0., 0., 1., 0., 0.])] * n_agents
    update_values = [False] * n_agents

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    policy = AttentionSACPolicy(n_agents, state_size, action_size, train_params)
    policy.prep_training('gpu' if train_params.use_gpu else 'cpu')

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print("\n🛑 Could't load replay buffer, were the experiences generated using the same tree depth?")
            print(e)
            exit(1)

    print("\n💾 Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params.buffer_size))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print("⚠️  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left.".format(hdd.free / (2 ** 30)))

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(train_env_params), {})
    writer.add_hparams(vars(obs_params), {})

    training_timer = Timer()
    training_timer.start()

    print("\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
        train_env.get_num_agents(),
        x_dim, y_dim,
        n_episodes,
        n_eval_episodes,
        checkpoint_interval,
        training_id
    ))

    make_dir(CHECKPOINT_DIR)
    params_file = os.path.join(CHECKPOINT_DIR, 'params.txt')
    write_params_to_file(train_params, train_env_params, obs_params, params_file)

    score_list = []
    completion_list = []

    for episode_idx in range(n_episodes + 1):
        # print('episode = ' + str(episode_idx))
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        if train_params.render:
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            # print('step = ' + str(step))
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], agent)

                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if train_params.render and episode_idx % checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            policy.step(agent_prev_obs, agent_prev_action, all_rewards, agent_obs, done)

            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    action_id = action_dict[agent]
                    agent_prev_action[agent] = np.array([0., 0., 0., 0., 0.])
                    agent_prev_action[agent][action_id] = 1

                # Preprocess the new observations
                if next_obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]

            nb_steps = step

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        score_list.append(smoothed_normalized_score)
        completion_list.append(smoothed_completion)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            # torch.save(policy.qnetwork_local, os.path.join(CHECKPOINT_DIR, str(episode_idx) + '.pth'))
            for agent_id in range(len(policy.agents)):
                torch.save(policy.agents[agent_id].policy, os.path.join(CHECKPOINT_DIR, str(episode_idx) + '_agent' + str(agent_id) + '.pth'))

            if save_replay_buffer:
                policy.save_replay_buffer('replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            if train_params.render:
                env_renderer.close_window()

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        # if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0:
        #     scores, completions, nb_steps_eval = eval_policy(eval_env, policy, train_params, obs_params)

        #     smoothing = 0.9
        #     smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
        #     smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)


    pickle_list(score_list, os.path.join(CHECKPOINT_DIR, 'scores.pkl'))
    pickle_list(completion_list, os.path.join(CHECKPOINT_DIR, 'completion.pkl'))

    plt.plot(score_list)
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'scores.png'))
    plt.show()
    
    plt.plot(completion_list)
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'completion.png'))
    plt.show()
    










def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, policy, train_params, obs_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        final_step = 0

        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(obs[agent], tree_depth=tree_depth, observation_radius=observation_radius)

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], agent)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps










def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_timestamp():
    ct = datetime.datetime.now()
    return str(ct).split('.')[0].replace(' ', '').replace('-', '').replace(':', '')

def pickle_list(l, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(l, f)

def write_params_to_file(train_params, train_env_params, obs_params, params_file):
    with open(params_file, "w") as file1:
        file1.write(f'n_episodes={train_params.n_episodes}' + '\n')
        file1.write(f'training_env_config={train_params.training_env_config}' + '\n')
        file1.write(f'evaluation_env_config={train_params.evaluation_env_config}' + '\n')
        file1.write(f'n_evaluation_episodes={train_params.n_evaluation_episodes}' + '\n')
        file1.write(f'checkpoint_interval={train_params.checkpoint_interval}' + '\n')
        file1.write(f'eps_start={train_params.eps_start}' + '\n')
        file1.write(f'eps_end={train_params.eps_end}' + '\n')
        file1.write(f'eps_decay={train_params.eps_decay}' + '\n')
        file1.write(f'buffer_size={train_params.buffer_size}' + '\n')
        file1.write(f'buffer_min_size={train_params.buffer_min_size}' + '\n')
        file1.write(f'restore_replay_buffer={train_params.restore_replay_buffer}' + '\n')
        file1.write(f'save_replay_buffer={train_params.save_replay_buffer}' + '\n')
        file1.write(f'batch_size={train_params.batch_size}' + '\n')
        file1.write(f'gamma={train_params.gamma}' + '\n')
        file1.write(f'tau={train_params.tau}' + '\n')
        file1.write(f'learning_rate={train_params.learning_rate}' + '\n')
        file1.write(f'hidden_size={train_params.hidden_size}' + '\n')
        file1.write(f'update_every={train_params.update_every}' + '\n')
        file1.write(f'use_gpu={train_params.use_gpu}' + '\n')
        file1.write(f'num_threads={train_params.num_threads}' + '\n')
        file1.write(f'render={train_params.render}' + '\n')
        file1.write(f'n_agents={train_env_params.n_agents}' + '\n')
        file1.write(f'x_dim={train_env_params.x_dim}' + '\n')
        file1.write(f'y_dim={train_env_params.y_dim}' + '\n')
        file1.write(f'n_cities={train_env_params.n_cities}' + '\n')
        file1.write(f'max_rails_between_cities={train_env_params.max_rails_between_cities}' + '\n')
        file1.write(f'max_rails_in_city={train_env_params.max_rails_in_city}' + '\n')
        file1.write(f'malfunction_rate={train_env_params.malfunction_rate}' + '\n')
        file1.write(f'seed={train_env_params.seed}' + '\n')
        file1.write(f'observation_tree_depth={obs_params.observation_tree_depth}' + '\n')
        file1.write(f'observation_radius={obs_params.observation_radius}' + '\n')
        file1.write(f'observation_max_path_depth={obs_params.observation_max_path_depth}' + '\n')










MSELoss = torch.nn.MSELoss()

# CHECKPOINT_DIR = '/scratch/ns4486/flatland-reinforcement-learning/single-agent/checkpoints'
CHECKPOINT_DIR = '.'
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, get_timestamp())

class Object(object):
    pass

training_params = Object()
training_params.n_episodes = 2500
training_params.training_env_config = 0
training_params.evaluation_env_config = 0
training_params.n_evaluation_episodes = 25
training_params.checkpoint_interval = 100
training_params.eps_start = 1.0
training_params.eps_end = 0.01
training_params.eps_decay = 0.99
training_params.buffer_size = int(1e5)
training_params.buffer_min_size = 0
training_params.restore_replay_buffer = ""
training_params.save_replay_buffer = False
training_params.batch_size = 128
training_params.gamma = 0.99
training_params.tau = 1e-3
training_params.learning_rate = 0.5e-4
training_params.hidden_size = 128
training_params.update_every = 8
training_params.use_gpu = True
training_params.num_threads = 1
training_params.render = False


env_params = [
    {
        # Test_0
        "n_agents": 2,
        "x_dim": 25,
        "y_dim": 25,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_rate": 1 / 50,
        "seed": 0
    },
    {
        # Test_1
        "n_agents": 10,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_rate": 1 / 100,
        "seed": 0
    },
    {
        # Test_2
        "n_agents": 20,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 3,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_rate": 1 / 200,
        "seed": 0
    },
]

obs_params = {
    "observation_tree_depth": 2,
    "observation_radius": 10,
    "observation_max_path_depth": 30
}

def check_env_config(id):
    if id >= len(env_params) or id < 0:
        print("\n🛑 Invalid environment configuration, only Test_0 to Test_{} are supported.".format(len(env_params) - 1))
        exit(1)


check_env_config(training_params.training_env_config)
check_env_config(training_params.evaluation_env_config)

training_env_params = env_params[training_params.training_env_config]
evaluation_env_params = env_params[training_params.evaluation_env_config]

print("\nTraining parameters:")
pprint(vars(training_params))
print("\nTraining environment parameters (Test_{}):".format(training_params.training_env_config))
pprint(training_env_params)
print("\nEvaluation environment parameters (Test_{}):".format(training_params.evaluation_env_config))
pprint(evaluation_env_params)
print("\nObservation parameters:")
pprint(obs_params)

os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params), Namespace(**obs_params))





