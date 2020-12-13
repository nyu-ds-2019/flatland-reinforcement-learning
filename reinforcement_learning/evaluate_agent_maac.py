import math
import multiprocessing
import os
import sys
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable

from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

import PIL
from PIL import Image

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.deadlock_check import check_if_all_blocked
from utils.timer import Timer
from utils.observation_utils import normalize_observation




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

        # hard_update(self.target_policy, self.policy)
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


def act(agent, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(torch.device('cpu'))
    action_values = agent.step(state, explore = True)
    return np.argmax(action_values.cpu().data.numpy())


def eval_policy(env_params, checkpoint, n_eval_episodes, max_steps, action_size, state_size, seed, render, allow_skipping, allow_caching):
    # Evaluation is faster on CPU (except if you use a really huge policy)
    parameters = {
        'use_gpu': False
    }

    # policy = DDDQNPolicy(state_size, action_size, Namespace(**parameters), evaluation_mode=True)
    # policy.qnetwork_local = torch.load(checkpoint, map_location={'cuda:0': 'cpu'})

    env_params = Namespace(**env_params)

    # Environment parameters
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city

    agents = []
    for agent_id in range(n_agents):
        agent = AttentionAgent(
            num_in_pol = state_size,
            num_out_pol = action_size,
            hidden_dim = 256,
            lr = 0.001
        )

        agent.policy = torch.load(os.path.join(checkpoint, f'2300_agent{agent_id}' + '.pth'), map_location=torch.device('cpu'))
        agent.policy.eval()

        agents.append(agent)

    # Malfunction and speed profiles
    # TODO pass these parameters properly from main!
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1. / 2000,  # Rate of malfunctions
        min_duration=20,  # Minimal duration
        max_duration=50  # Max duration
    )

    # Only fast trains in Round 1
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_radius = env_params.observation_radius
    observation_max_path_depth = env_params.observation_max_path_depth

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environment
    env = RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city,
        ),
        # rail_generator = complex_rail_generator(
        #     nr_start_goal=10,
        #     nr_extra=10,
        #     min_dist=10,
        #     max_dist=99999,
        #     seed=1
        # ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=tree_observation
    )

    if render:
        # env_renderer = RenderTool(env, gl="PGL")
        env_renderer = RenderTool(
            env, 
            # gl="PGL",
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=False,
            screen_height=600,  # Adjust these parameters to fit your resolution
            screen_width=800
        )

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []
    inference_times = []
    preproc_times = []
    agent_times = []
    step_times = []

    for agent_id in range(n_agents):
        action_dict[agent_id] = 0

    for episode_idx in range(n_eval_episodes):
        images = []
        seed += 1

        inference_timer = Timer()
        preproc_timer = Timer()
        agent_timer = Timer()
        step_timer = Timer()

        step_timer.start()
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
        step_timer.end()

        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        if render:
            env_renderer.set_new_rail()

        final_step = 0
        skipped = 0

        nb_hit = 0
        agent_last_obs = {}
        agent_last_action = {}

        for step in range(max_steps - 1):
            # time.sleep(0.2)
            if allow_skipping and check_if_all_blocked(env):
                # FIXME why -1? bug where all agents are "done" after max_steps!
                skipped = max_steps - step - 1
                final_step = max_steps - 2
                n_unfinished_agents = sum(not done[idx] for idx in env.get_agent_handles())
                score -= skipped * n_unfinished_agents
                break

            agent_timer.start()
            for agent in env.get_agent_handles():
                agent_model = agents[agent]
                if obs[agent] and info['action_required'][agent]:
                    if agent in agent_last_obs and np.all(agent_last_obs[agent] == obs[agent]):
                        nb_hit += 1
                        action = agent_last_action[agent]

                    else:
                        preproc_timer.start()
                        norm_obs = normalize_observation(obs[agent], tree_depth=observation_tree_depth, observation_radius=observation_radius)
                        preproc_timer.end()

                        inference_timer.start()
                        action = act(agent_model, norm_obs)
                        inference_timer.end()

                    action_dict.update({agent: action})

                    if allow_caching:
                        agent_last_obs[agent] = obs[agent]
                        agent_last_action[agent] = action
            agent_timer.end()

            step_timer.start()
            obs, all_rewards, done, info = env.step(action_dict)
            step_timer.end()

            if render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

                im = env_renderer.get_image()
                im = PIL.Image.fromarray(im)
                images.append(im)

                if step % 100 == 0:
                    print("{}/{}".format(step, max_steps - 1))

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        if render:
            for _ in range(10):
                images.append(images[len(images)-1])

            # save video
            images[0].save(f'/Users/nikhilvs/repos/nyu/flatland-reinforcement-learning/videos/maac-final/out_{episode_idx}.gif',
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=60,
                        loop=0)

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

        inference_times.append(inference_timer.get())
        preproc_times.append(preproc_timer.get())
        agent_times.append(agent_timer.get())
        step_times.append(step_timer.get())

        skipped_text = ""
        if skipped > 0:
            skipped_text = "\t⚡ Skipped {}".format(skipped)

        hit_text = ""
        if nb_hit > 0:
            hit_text = "\t⚡ Hit {} ({:.1f}%)".format(nb_hit, (100 * nb_hit) / (n_agents * final_step))

        print(
            "☑️  Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} "
            "\t🍭 Seed: {}"
            "\t🚉 Env: {:.3f}s  "
            "\t🤖 Agent: {:.3f}s (per step: {:.3f}s) \t[preproc: {:.3f}s \tinfer: {:.3f}s]"
            "{}{}".format(
                normalized_score,
                completion * 100.0,
                final_step,
                seed,
                step_timer.get(),
                agent_timer.get(),
                agent_timer.get() / final_step,
                preproc_timer.get(),
                inference_timer.get(),
                skipped_text,
                hit_text
            )
        )

    return scores, completions, nb_steps, agent_times, step_times


def evaluate_agents(file, n_evaluation_episodes, use_gpu, render, allow_skipping, allow_caching):
    nb_threads = 1
    eval_per_thread = n_evaluation_episodes

    if not render:
        nb_threads = multiprocessing.cpu_count()
        eval_per_thread = max(1, math.ceil(n_evaluation_episodes / nb_threads))

    total_nb_eval = eval_per_thread * nb_threads
    print("Will evaluate policy {} over {} episodes on {} threads.".format(file, total_nb_eval, nb_threads))

    if total_nb_eval != n_evaluation_episodes:
        print("(Rounding up from {} to fill all cores)".format(n_evaluation_episodes))

    # Observation parameters need to match the ones used during training!

    # small_v0
    small_v0_params = {
        # sample configuration
        "n_agents": 2,
        "x_dim": 25,
        "y_dim": 25,
        "n_cities": 2,
        "max_rails_between_cities": 3,
        "max_rails_in_city": 2,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }

    # Test_0
    test0_params = {
        # sample configuration
        "n_agents": 5,
        "x_dim": 25,
        "y_dim": 25,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 20
    }

    # Test_1
    test1_params = {
        # environment
        "n_agents": 10,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 10
    }

    # Test_5
    test5_params = {
        # environment
        "n_agents": 80,
        "x_dim": 35,
        "y_dim": 35,
        "n_cities": 5,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 4,

        # observations
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 20
    }

    params = small_v0_params
    env_params = Namespace(**params)

    print("Environment parameters:")
    pprint(params)

    # Calculate space dimensions and max steps
    max_steps = int(4 * 2 * (env_params.x_dim + env_params.y_dim + (env_params.n_agents / env_params.n_cities)))
    action_size = 5
    tree_observation = TreeObsForRailEnv(max_depth=env_params.observation_tree_depth)
    tree_depth = env_params.observation_tree_depth
    num_features_per_node = tree_observation.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(tree_depth + 1)])
    state_size = num_features_per_node * n_nodes

    results = []
    if render:
        results.append(eval_policy(params, file, eval_per_thread, max_steps, action_size, state_size, 0, render, allow_skipping, allow_caching))

    else:
        with Pool() as p:
            results = p.starmap(eval_policy,
                                [(params, file, 1, max_steps, action_size, state_size, seed * nb_threads, render, allow_skipping, allow_caching)
                                 for seed in
                                 range(total_nb_eval)])

    scores = []
    completions = []
    nb_steps = []
    times = []
    step_times = []
    for s, c, n, t, st in results:
        scores.append(s)
        completions.append(c)
        nb_steps.append(n)
        times.append(t)
        step_times.append(st)

    print("-" * 200)

    print("✅ Score: {:.3f} \tDone: {:.1f}% \tNb steps: {:.3f} \tAgent total: {:.3f}s (per step: {:.3f}s)".format(
        np.mean(scores),
        np.mean(completions) * 100.0,
        np.mean(nb_steps),
        np.mean(times),
        np.mean(times) / np.mean(nb_steps)
    ))

    print("⏲️  Agent sum: {:.3f}s \tEnv sum: {:.3f}s \tTotal sum: {:.3f}s".format(
        np.sum(times),
        np.sum(step_times),
        np.sum(times) + np.sum(step_times)
    ))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="checkpoint to load", required=True, type=str)
    parser.add_argument("-n", "--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)

    # TODO
    # parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0, type=int)

    parser.add_argument("--use_gpu", dest="use_gpu", help="use GPU if available", action='store_true')
    parser.add_argument("--render", help="render a single episode", action='store_true')
    parser.add_argument("--allow_skipping", help="skips to the end of the episode if all agents are deadlocked", action='store_true')
    parser.add_argument("--allow_caching", help="caches the last observation-action pair", action='store_true')
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(1)
    evaluate_agents(file=args.file, n_evaluation_episodes=args.n_evaluation_episodes, use_gpu=args.use_gpu, render=args.render,
                    allow_skipping=args.allow_skipping, allow_caching=args.allow_caching)
