from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import itertools

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch, meta_reset, prev_hid):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()

        nagents = state.size()[1]

        prev_action = np.array([0] * nagents)
        prev_reward = [0] * nagents

        for t in range(self.args.max_steps):
            misc = dict()
            # recurrence over time
            if  meta_reset and t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0], nagents=nagents)

            if self.args.vanilla:
                x = [state, prev_hid]
            else:
                x = [state, prev_action, prev_reward, prev_hid]
            action_out, value, prev_hid = self.policy_net(x, info)

            if self.args.vanilla:
                if (t + 1) % self.args.detach_gap == 0:
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)
            prev_action = action[0]
            prev_reward = reward

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            stat['task%i_reward' % (self.env.env.cur_idx)] = stat.get('task%i_reward' % (self.env.env.cur_idx), 0) + reward[:nagents]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break

        stat['task%i_num_steps' % (self.env.env.cur_idx)] = t + 1
        stat['task%i_steps_taken' % (self.env.env.cur_idx)] = stat['task%i_num_steps' % (self.env.env.cur_idx)]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat, prev_hid)


    def compute_grad(self, batch):
        stat = dict()
        # action space
        num_actions = self.args.num_actions
        # number of action head
        dim_actions = self.args.dim_actions

        n = self.env.env.cur_nagents
        batch_size = len(batch.state)

        # size: batch_size * n
        rewards = torch.Tensor(batch.reward)
        # size: batch_size * n
        episode_masks = torch.Tensor(batch.episode_mask)
        # size: batch_size * n
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        # size: batch_size * n * dim_actions.  have been detached
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        # size: (batch_size*n)
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()
            
        # element size: (batch_size*n) * num_actions[i]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # size: (batch_size*n) * dim_actions
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # size: (batch_size*n) * dim_actions
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            # size: (batch_size*n) * 1
            log_prob = multinomials_log_density(actions, log_p_a)
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        # stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        # stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        # entropy regularization term
        entropy = 0
        for i in range(len(log_p_a)):
            entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
        # stat['entropy'] = entropy.item()
        if self.args.entr > 0:
            loss -= self.args.entr * entropy

        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['task%i_num_episodes' % (self.env.env.cur_idx)] = 0
        count = 0
        meta_reset = False
        hid_state = None
        while len(batch) < self.args.batch_size:
            count += 1
            if self.args.vanilla:
                meta_reset = True
            else:
                if count == 1:
                    meta_reset = True
                else:
                    meta_reset = False
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat, hid_state = self.get_episode(epoch, meta_reset, hid_state)
            merge_stat(episode_stat, self.stats)
            self.stats['task%i_num_episodes' % (self.env.env.cur_idx)] += 1
            batch += episode

        self.last_step = False
        self.stats['task%i_num_steps' % (self.env.env.cur_idx)] = len(batch)
        # print(self.stats['task%i_num_steps' % (self.env.env.cur_idx)])
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()
        if self.args.run_mode != "test":
            s = self.compute_grad(batch)
            merge_stat(s, stat)
            for p in self.params:
                if p._grad is not None:
                    p._grad.data /= stat['task%i_num_steps' % (self.env.env.cur_idx)]
            self.optimizer.step()
        self.env.change_env()
        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)