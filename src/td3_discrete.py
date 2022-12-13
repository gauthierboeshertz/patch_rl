from .td3_utils import weight_init, TruncatedNormal, schedule, to_torch, soft_update_params
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from operator import itemgetter
import einops

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class MultiDiscreteActor(nn.Module):
    def __init__(self, obs_dim, num_actions,action_dim, hidden_dim):
        super().__init__()

        print("num_actions",num_actions)
        print("action_dim",action_dim)
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.policy_trunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    )

        self.policy_heads = nn.ModuleList([nn.Linear(hidden_dim, num_actions,bias=False) for i in range(self.action_dim)])
        self.apply(weight_init)
        print("self.policy_heads",self.policy_heads)
        
    def forward(self, obs, std=None,sample=False):
        
        trunk = self.policy_trunk(obs)
        logits = torch.stack([ head(trunk) for head in self.policy_heads],dim=1)
        
        if sample:
            #logits +=  torch.normal(mean=torch.zeros_like(logits), std=torch.ones_like(logits)*std)
            act = torch.stack([ F.gumbel_softmax(logits=logits[:,i],hard=True,tau=2) for i in range(self.action_dim) ],dim=1)
            act = einops.rearrange(act,'b n a -> b (n a)')
        else:
            act = F.softmax(logits, dim=-1) #torch.stack([ F.softmax(logits[:,i],dim=-1) for i in range(self.action_dim) ],dim=1)
        return act     
        
        

class Critic(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + num_actions, hidden_dim),
            nn.LayerNorm(hidden_dim),nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + num_actions, hidden_dim),
            nn.LayerNorm(hidden_dim),nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(weight_init)

    def forward(self, obs, action):
        #action = F.one_hot(action.long(), num_classes=2)#einops.rearrange(action,'b n -> b n a',n=1)
        #action = einops.rearrange(action,'b n a -> b (n a)')

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)

        return q1, q2


class TD3Agent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 num_actions,
                 device,
                 lr,
                 hidden_dim,
                 critic_target_tau,
                 stddev_schedule,
                 stddev_clip,
                 use_tb):
        self.action_dim = action_shape[0]
        self.num_actions = num_actions
        print("action_dim",self.action_dim)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.discount = 0.9
        # models
        self.actor = MultiDiscreteActor(obs_shape[0], num_actions,action_shape[0],
                           hidden_dim).to(device)

        self.critic = Critic(obs_shape[0], num_actions *self.action_dim ,
                             hidden_dim).to(device)
        self.critic_target = Critic(obs_shape[0], num_actions*self.action_dim,
                                    hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode=False):
        stddev = schedule(self.stddev_schedule, step)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        action = self.actor(obs, std = stddev,sample=not eval_mode)
        action = action.reshape(self.action_dim,self.num_actions)
        action = action.argmax(dim=-1)
        action = action.cpu().numpy()
        #if action.shape[0] == 1:
        #    action = action[0]
        return np.array(action)

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = schedule(self.stddev_schedule, step)
            next_action = self.actor(next_obs, std=stddev,sample=True)#dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        action = F.one_hot(action.long(), num_classes=self.num_actions)
        action = einops.rearrange(action,'b n a -> b (n a)')
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)
        action_sampled =self.actor(obs, std=stddev,sample=True) #policy = self.actor(obs, stddev)

        Q1, Q2 = self.critic(obs, action_sampled)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            #metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, batch, step):
        metrics = dict()

        #batch = next(replay_iter)
        batch = to_torch(
            batch, self.device)
        obs, action, next_obs, reward = itemgetter('obs', 'action',"next_obs","reward")(batch)
        discount = torch.ones_like(reward).to(reward.device) * self.discount
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
