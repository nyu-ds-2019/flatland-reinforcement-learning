
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var

# from logger import Logger
from env_utils import normalize_observation

import sys

# Set the logger
# logger = Logger('./logs') # dive in later
# step=0

def to_np(x): # from tensor to numpy
    return x.data.cpu().numpy()

def to_var(x): # from tensor to Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class MAA2C(Agent):
    """
    An multi-agent learned with Advantage Actor-Critic
    - Actor takes its local observations as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy

    Parameters
    - training_strategy:
        - cocurrent
            - each agent learns its own individual policy which is independent
            - multiple policies are optimized simultaneously
        - centralized (see MADDPG in [1] for details)
            - centralized training and decentralized execution
            - decentralized actor map it's local observations to action using individual policy
            - centralized critic takes both state and action from all agents as input, each actor
                has its own critic for estimating the value function, which allows each actor has
                different reward structure, e.g., cooperative, competitive, mixed task
    - actor_parameter_sharing:
        - True: all actors share a single policy which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous. Please see Sec. 4.3 in [2] and
            Sec. 4.1 & 4.2 in [3] for details.
        - False: each actor use independent policy
    - critic_parameter_sharing:
        - True: all actors share a single critic which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous and reward sharing holds. Please
            see Sec. 4.1 in [3] for details.
        - False: each actor use independent critic (though each critic can take other agents actions
            as input, see MADDPG in [1] for details)

    Reference:
    [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
    [2] Cooperative Multi-Agent Control Using Deep Reinforcement Learning
    [3] Parameter Sharing Deep Deterministic Policy Gradient for Cooperative Multi-agent Reinforcement Learning

    """
    def __init__(self, env, n_agents, obs_shape_n, act_shape_n,
                 memory_capacity=int(1e5), max_steps=100,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=256, critic_hidden_size=256,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=3e-4, critic_lr=3e-4,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=1, batch_size=128, episodes_before_train=1,
                 epsilon_start=0.98, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True, training_strategy="centralized",
                 actor_parameter_sharing=False, critic_parameter_sharing=False):
        super(MAA2C, self).__init__(env, obs_shape_n, act_shape_n,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        assert training_strategy in ["cocurrent", "centralized"]

        self.n_agents = n_agents
        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing

        # self.actors = [ActorNetwork(self.obs_shape_n, self.actor_hidden_size, self.act_shape_n, self.actor_output_act)] * self.n_agents
        self.actors = []
        for i in range(self.n_agents):
            self.actors.append(ActorNetwork(self.obs_shape_n, self.actor_hidden_size, self.act_shape_n, self.actor_output_act))

        self.critics = []
        self.whole_critic_state_dim = 0
        self.whole_critic_action_dim = 0
        if self.training_strategy == "cocurrent":
            for i in range(self.n_agents):
                self.critics.append(CriticNetwork(self.obs_shape_n, self.act_shape_n, self.critic_hidden_size, 1))
            # self.critics = [CriticNetwork(self.obs_shape_n, self.act_shape_n, self.critic_hidden_size, 1)] * self.n_agents
        elif self.training_strategy == "centralized":
            for i in range(self.n_agents):
                self.whole_critic_state_dim += self.obs_shape_n
                self.whole_critic_action_dim += self.act_shape_n
            for i in range(self.n_agents):
                self.critics.append(CriticNetwork(self.whole_critic_state_dim, self.whole_critic_action_dim, self.critic_hidden_size, 1))
            # critic_state_dim = self.n_agents * self.obs_shape_n
            # critic_action_dim = self.n_agents * self.act_shape_n
            # self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, self.critic_hidden_size, 1)] * self.n_agents
            # print(whole_critic_state_dim, whole_critic_action_dim)

        if optimizer_type == "adam":
            self.actor_optimizers = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizers = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizers = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        # tricky and memory consumed implementation of parameter sharing
        if self.actor_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.actors[agent_id] = self.actors[0]
                self.actor_optimizers[agent_id] = self.actor_optimizers[0]
        if self.critic_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.critics[agent_id] = self.critics[0]
                self.critic_optimizers[agent_id] = self.critic_optimizers[0]

        if self.use_cuda:
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()

    def _normalize_state(self, obs):
        observation_tree_depth = 3
        observation_radius = 30

        agent_obs = [None] * self.env.get_num_agents()

        for agent in self.env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius)
                # agent_prev_obs[agent] = agent_obs[agent].copy()

        return agent_obs

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, self.info_state = self.env.reset()
            self.env_state = self._normalize_state(self.env_state)
            self.n_steps = 0

        states = []
        actions = []
        rewards = []
        terminal = False

        if type(self.env_state) is dict:
            self.env_state = self._normalize_state(self.env_state)
        
        # take n steps
        for i in range(self.roll_out_n_steps):
            assert type(self.env_state) is not dict,  f"{self.n_steps}"
            states.append(self.env_state)
            action = self.exploration_action(self.env_state)
            next_state, reward, done, info = self.env.step(action)
            # next_state = self._normalize_state(next_state)

            agent_obs = [None] * self.n_agents
            for agent in self.env.get_agent_handles():
                if not info['action_required'][agent] or done[agent]:
                    agent_obs[agent] = self.env_state[agent]
                    continue
                if next_state[agent]:
                    agent_obs[agent] = normalize_observation(next_state[agent], 3, 30)
                else:
                    agent_obs[agent] = self.env_state[agent]
            
            next_state = agent_obs

            done = done['__all__']
            # actions.append([index_to_one_hot(a, self.act_shape_n) for a in action])
            actions.append([index_to_one_hot(a, self.act_shape_n) for a in action])

            # print(reward)
            rewards.append(reward)
            final_state = next_state

            self.env_state = next_state

            if done:
                self.env_state, self.info_state = self.env.reset()
                self.env_state = self._normalize_state(self.env_state)
                self.n_steps = 0
                break

        # for displaying learned policies
        # time.sleep(0.1)
        # self.env.render()
        # discount reward
        if done:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            # print("done")
        else:
            one_hot_action = []
            self.episode_done = False
            final_action = self.action(final_state)
            one_hot_action = [index_to_one_hot(a, self.act_shape_n) for a in final_action]
            final_r = self.value(final_state, one_hot_action)

        new_rewards = []
        for reward in rewards:
            new_reward = []
            for agent_id in range(self.n_agents):
                new_reward.append(reward[agent_id])
            new_rewards.append(new_reward)

        rewards = np.array(new_rewards)

        for agent_id in range(self.n_agents):
            rewards[:,agent_id] = self._discount_reward(rewards[:,agent_id], final_r[agent_id])

        rewards = rewards.tolist()
        # print(rewards)
        self.n_steps += 1
        self.memory.push(states, actions, rewards)

        # print(actions)
        # print(np.mean(rewards))

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.obs_shape_n)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.act_shape_n)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        whole_states_var = states_var.view(-1, self.whole_critic_state_dim)
        whole_actions_var = actions_var.view(-1, self.whole_critic_action_dim)

        # print( states_var )
        # print(self.n_agents)
        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizers[agent_id].zero_grad()
            # print(states_var[:,agent_id,:])
            action_log_probs = self.actors[agent_id](states_var[:,agent_id,:])
            entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
            action_log_probs = th.sum(action_log_probs * actions_var[:,agent_id,:], 1)
            if self.training_strategy == "cocurrent":
                values = self.critics[agent_id](states_var[:,agent_id,:], actions_var[:,agent_id,:])
            elif self.training_strategy == "centralized":
                values = self.critics[agent_id](whole_states_var, whole_actions_var)
            advantages = rewards_var[:,agent_id,:] - values.detach()
            pg_loss = -th.mean(action_log_probs * advantages)
            actor_loss = pg_loss - entropy_loss * self.entropy_reg
            # actor_loss = pg_loss
            actor_loss.backward()
            # print(self.actors[agent_id].parameters())
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent_id].step()

            # update critic network
            self.critic_optimizers[agent_id].zero_grad()
            target_values = rewards_var[:,agent_id,:]
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critic_optimizers[agent_id].step()


    # predict softmax action based on state
    def _softmax_action(self, state):
        try:
            state_var = to_tensor_var([state], self.use_cuda)
        except ValueError as e:
            print([state])
            sys.exit(0)
        softmax_action = np.zeros((self.n_agents, self.act_shape_n), dtype=np.float64)
        for agent_id in range(self.n_agents):
            softmax_action_var = th.exp(self.actors[agent_id](state_var[:,agent_id,:]))
            if self.use_cuda:
                softmax_action[agent_id] = softmax_action_var.data.cpu().numpy()[0]
            else:
                softmax_action[agent_id] = softmax_action_var.data.numpy()[0]
        return softmax_action

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        actions = {}
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                     np.exp(-1. * self.n_steps / self.epsilon_decay)
        for agent_id in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[agent_id] = np.random.choice(self.act_shape_n)
            else:
                actions[agent_id] = np.argmax(softmax_action[agent_id])
        return actions

    # predict action based on state for execution
    def action(self, state):
        softmax_actions = self._softmax_action(state)
        actions = np.argmax(softmax_actions, axis=1)
        return actions

    # evaluate value
    def value(self, state, action):
        # print([state])
        # try:
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        # except ValueError as e:
        #     print([state])
        #     print([action])
        whole_state_var = state_var.view(-1, self.n_agents*self.obs_shape_n)
        whole_action_var = action_var.view(-1, self.n_agents*self.act_shape_n)
        values = [0]*self.n_agents
        for agent_id in range(self.n_agents):
            if self.training_strategy == "cocurrent":
                value_var = self.critics[agent_id](state_var[:,agent_id,:], action_var[:,agent_id,:])
            elif self.training_strategy == "centralized":
                value_var = self.critics[agent_id](whole_state_var, whole_action_var)
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    def evaluation(self, env, eval_episodes=10):

        for agent_id in range(self.n_agents):
            self.actors[agent_id].eval()
        with th.no_grad():
            
            rewards = []
            infos = []
            for i in range(eval_episodes):
                rewards_i = []
                infos_i = []

                state, info = env.reset()
                state = self._normalize_state(state)

                softmax_action = self.action(state)

                action = {}

                for agent_id in range(self.n_agents):
                    action[agent_id] = softmax_action[agent_id]

                state, rew, done, info = env.step(action)
                state = self._normalize_state(state)

                reward = []
                for agent_id in range(self.n_agents):
                    reward.append(rew[agent_id])

                done = done['__all__']
                rewards_i.append(sum(reward))
                infos_i.append(info)
                while not done:
                    softmax_action = self.action(state)
                    action = {}
                    for agent_id in range(self.n_agents):
                        action[agent_id] = softmax_action[agent_id]

                    next_state, rew, done, info = env.step(action)

                    agent_obs = [None] * self.n_agents
                    for agent in self.env.get_agent_handles():
                        if not info['action_required'][agent] or done[agent]:
                            agent_obs[agent] = state[agent]
                            continue
                        if next_state[agent]:
                            agent_obs[agent] = normalize_observation(next_state[agent], 3, 30)
                        else:
                            agent_obs[agent] = state[agent]
                    
                    next_state = agent_obs  

                    state = next_state

                    # state = self._normalize_state(state)

                    reward = []
                    for agent_id in range(self.n_agents):
                        reward.append(rew[agent_id])

                    done = done['__all__']
                    rewards_i.append(sum(reward))
                    infos_i.append(info)

                rewards.append(rewards_i)
                infos.append(infos_i)
                
        for agent_id in range(self.n_agents):
            self.actors[agent_id].train()
        return rewards, infos