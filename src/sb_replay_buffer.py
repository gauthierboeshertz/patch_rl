import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
import torch
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)


from .patch_utils import image_to_patches, patches_to_image
from .mask_utils import get_ccs_to_swap, swap_transition_components, get_cc_dicts_from_mask


class CodaPatchReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        mask_function = None,
        num_patches = None,
        patch_size = None,
        coda_minimum_size = None,
        coda_buffer_size= None,
        coda_batch_ratio= None,
        group_actions = True,
        reward_function = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,optimize_memory_usage=optimize_memory_usage,handle_timeout_termination=handle_timeout_termination)

        self.mask_function = mask_function
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.coda_buffer_size = coda_buffer_size
        self.num_sample_iterations = 0
        self.coda_minimum_size = coda_minimum_size
        self.coda_batch_ratio = coda_batch_ratio
        self.has_coda_buffer = False
        self.group_actions = group_actions
        self.reward_function = reward_function
        self.last_coda_pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        self.num_sample_iterations += 1
        upper_bound = self.buffer_size if self.full else self.pos
        
        if self.coda_minimum_size > 0 and (((self.pos % self.coda_minimum_size) == 0) and self.pos > 0 and self.pos != self.last_coda_pos):
            self.do_coda_all()
            self.has_coda_buffer = True
            self.last_coda_pos = self.pos
        
        if self.has_coda_buffer:
            batch_inds = np.random.randint(0, upper_bound, size=int((1-self.coda_batch_ratio)* batch_size))
            batch_data = self._get_samples(batch_inds, env)
            
            coda_batch_inds = np.random.randint(0, self.coda_observations.shape[0], size=int(self.coda_batch_ratio* batch_size))
            coda_batch_data = self._get_samples(coda_batch_inds, env,coda_samples=True)

            batch_states = np.concatenate((batch_data[0],coda_batch_data[0]),0)
            batch_actions = np.concatenate((batch_data[1],coda_batch_data[1]),0)
            batch_next_states = np.concatenate((batch_data[2],coda_batch_data[2]),0)
            batch_dones = np.concatenate((batch_data[3],coda_batch_data[3]),0)
            batch_rewards = np.concatenate((batch_data[4],coda_batch_data[4]),0)

            batch_data = (batch_states,batch_actions,batch_next_states,batch_dones,batch_rewards)
            
            """

            print("coda batch ratio: ", self.coda_batch_ratio)
            print("batch size: ", batch_size)
            print("coda batch size: ", int(self.coda_batch_ratio* batch_size))
            print("normal rewards: ", batch_data.rewards.shape)
            print("normal rewards sum: ", batch_data.rewards.sum())
            print("coda rewards: ", coda_batch_data.rewards.shape)
            print("coda rewards sum: ", coda_batch_data.rewards.sum())
            self.save_batch(batch_data,"")
            self.save_batch(coda_batch_data,"coda_")
            quit()
            batch_states = torch.cat((batch_data.observations,coda_batch_data.observations),0)
            batch_actions = torch.cat((batch_data.actions,coda_batch_data.actions),0)
            batch_next_states = torch.cat((batch_data.next_observations,coda_batch_data.next_observations),0)
            batch_rewards = torch.cat((batch_data.rewards,coda_batch_data.rewards),0)
            batch_dones = torch.cat((batch_data.dones,coda_batch_data.dones),0)
            return ReplayBufferSamples(batch_states, batch_actions, batch_next_states,batch_rewards, batch_dones)
            """
        else:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            batch_data = self._get_samples(batch_inds, env)
        return self._give_batch(batch_data, env)
    
    
    def _give_batch(self,batch,env):
        data = (
            self._normalize_obs(batch[0], env),
            batch[1],
            self._normalize_obs(batch[2], env),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            batch[3],
            self._normalize_reward(batch[4], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

        
        
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None, coda_samples= False) -> ReplayBufferSamples:
        # Sample randomly the env idx
        if coda_samples:
            observations = self.coda_observations
            next_observations = self.coda_next_observations
            actions = self.coda_actions
            rewards = self.coda_rewards
            dones = self.coda_dones
            #timeouts = self.coda_dones
        else:
            observations = self.observations
            next_observations = self.next_observations
            actions = self.actions
            rewards = self.rewards
            dones = self.dones
            #timeouts = self.timeouts
        
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        timeouts = dones * (rewards <= 0)
        
        return observations[batch_inds, env_indices, :], actions[batch_inds, env_indices, :], next_observations[batch_inds, env_indices, :], (dones[batch_inds, env_indices] * (1 - timeouts[batch_inds, env_indices])).reshape(-1, 1), rewards[batch_inds, env_indices].reshape(-1, 1)
        """
        data = (
            self._normalize_obs(observations[batch_inds, env_indices, :], env),
            actions[batch_inds, env_indices, :],
            self._normalize_obs(next_observations[batch_inds, env_indices, :], env),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (dones[batch_inds, env_indices] * (1 - timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        """
    def save_batch(self,batch,name_prefix):
        
        for field, value in zip(batch._fields, batch):
            torch.save(value, name_prefix + field + ".pt")

    def get_transition_data(self, idx):
        obs = torch.from_numpy(self.observations[idx,0])
        action = torch.from_numpy(self.actions[idx,0])
        next_obs = torch.from_numpy(self.next_observations[idx,0])
        reward = (self.rewards[idx,0])
        done = (self.dones[idx,0])
        ccs = self.ccs[idx]
        return obs, action, next_obs, reward, done, ccs
    
    def can_do_coda(self, transition1, transition2):
        _, _, _, _, done1, _ = transition1 
        _, _, _, _, done2, _ = transition2

        return  done1 == 0 and done2 == 0
    
    def do_coda_on_transitions(self, transition1, transition2):
        
        image1, action1, next_image1, reward1, done1, ccs1 = transition1   
        image2, action2, next_image2, reward2, done2, ccs2 = transition2
        
            
        patches1 = image_to_patches(image1, patch_size=self.patch_size,num_patches_sqrt=image1.shape[2]//self.patch_size[0])
        patches2 = image_to_patches(image2, patch_size=self.patch_size,num_patches_sqrt=image2.shape[2]//self.patch_size[0])

        next_patches1 = image_to_patches(next_image1, patch_size=self.patch_size,num_patches_sqrt=image1.shape[2]//self.patch_size[0])
        next_patches2 = image_to_patches(next_image2, patch_size=self.patch_size,num_patches_sqrt=image2.shape[2]//self.patch_size[0])

        comp_to_swap_ccs1, comp_to_swap_ccs2 = get_ccs_to_swap(ccs1, ccs2)
        
        swapped_transitions = swap_transition_components(patches1,patches2, action1,action2, next_patches1, next_patches2, 
                                            comp_to_swap_ccs1,comp_to_swap_ccs2,group_actions=self.group_actions)
        
        if swapped_transitions:
            swapped_patches = patches_to_image(torch.stack(swapped_transitions[0]), self.patch_size[0], image1.shape[-2])
            swapped_next_patches = patches_to_image(torch.stack(swapped_transitions[2]), self.patch_size[0], image1.shape[-2])
            
            return swapped_patches, swapped_transitions[1], swapped_next_patches
        else:
            return None

    def make_all_ccs(self):
        self.ccs = []
        batch_size = 100
        
        for i in range(0,self.observations.shape[0],batch_size):
            obs = self.to_torch(self.observations[i:i+batch_size,0])
            next_obs = self.to_torch(self.next_observations[i:i+batch_size,0])
            actions = self.to_torch(self.actions[i:i+batch_size,0])
            masks = self.mask_function(obs,actions,next_obs)
            ccss = [get_cc_dicts_from_mask(mask,num_patches=64,num_actions= actions.shape[-1]//(2 if self.group_actions else 1))  for mask in masks]
            self.ccs.extend(ccss)
        
        
    def do_coda_all(self):
        
        print("Doing coda on all transitions")
        
        coda_obs = []
        coda_obs2 = []
        coda_act = []
        
        with torch.no_grad():
            self.make_all_ccs()
        
        upper_bound = self.buffer_size if self.full else self.pos
        
        while len(coda_obs) < self.coda_buffer_size:
            transition_idxs = np.random.choice( np.arange(max(self.pos - self.coda_minimum_size,0),upper_bound), 2, replace=False)
            transition1 = self.get_transition_data(transition_idxs[0])
            transition2 = self.get_transition_data(transition_idxs[1])
            
            if not self.can_do_coda(transition1, transition2): continue
            
            coda_transitions = self.do_coda_on_transitions(transition1, transition2)
            if coda_transitions is not None:
                coda_obs.extend(coda_transitions[0])
                coda_act.extend(coda_transitions[1])
                coda_obs2.extend(coda_transitions[2])
                
        coda_obs = torch.stack(coda_obs)
        coda_obs2 = torch.stack(coda_obs2)
        coda_act = torch.stack(coda_act)
        bs_size = 100
        coda_rewards = []
        for i in range(0,coda_obs.shape[0],bs_size):
            max_size = min(i + bs_size,coda_obs.shape[0])
            new_coda_rewards = self.reward_function(coda_obs[i:max_size],coda_act[i:max_size],coda_obs2[i:max_size])
            coda_rewards.append(new_coda_rewards)
        
        coda_rewards = torch.cat(coda_rewards)
        print("Rewards:",coda_rewards.shape)
        print("rewards sums",coda_rewards.sum())
        #coda_rewards =  torch.cat([self.reward_function(coda_obs[i:i+bs_size],coda_act[i:i+bs_size],coda_obs2[i:i+bs_size]) for i in range(0,coda_obs.shape[0],bs_size)],dim=0)
        self.coda_observations = coda_obs.unsqueeze(1).cpu().numpy()
        self.coda_next_observations = coda_obs2.unsqueeze(1).cpu().numpy()
        self.coda_actions = coda_act.unsqueeze(1).cpu().numpy()
        self.coda_rewards = coda_rewards.unsqueeze(1).cpu().numpy()
        self.coda_dones = torch.zeros((self.coda_observations.shape[0],1)).unsqueeze(1).cpu().numpy()
        print("Coda obs shape: ", self.coda_observations.shape)
