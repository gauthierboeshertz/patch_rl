import numpy as np
import torch
from .mask_utils import do_coda_on_transitions,get_cc_dicts_from_mask

class CodaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mask_function, reward_function, max_coda_transitions=-1,
                 patch_size=(16,16), num_actions=4,num_patches=64):
        
        self.mask_function = mask_function
        self.reward_function = reward_function
        self.max_coda_transitions = max_coda_transitions
        self.patch_size = patch_size
        self.num_actions = num_actions
        self.num_patches = num_patches
        self._create_all_ccs(dataset)
        coda_observations, coda_actions, coda_next_observations, coda_rewards, coda_dones = self._do_coda(dataset)
        
        self.observations = torch.cat((dataset.observations, coda_observations), dim=0).type(torch.uint8)
        self.actions = torch.cat((dataset.actions, coda_actions), dim=0)
        self.next_observations = torch.cat((dataset.next_observations, coda_next_observations), dim=0).type(torch.uint8)
        self.rewards = torch.cat((dataset.rewards, coda_rewards), dim=0)
        self.dones = torch.cat((dataset.dones, coda_dones), dim=0)
        
        
    
    def _create_all_ccs(self,dataset):
        masks = []
        batch_size = 100
        for i in range(0,len(dataset),batch_size):
            max_idx = min(i+batch_size,len(dataset))
            obs = dataset.observations[i:i+max_idx]
            action = dataset.actions[i:i+max_idx]
            next_obs = dataset.next_observations[i:i+max_idx]
            mask = self.mask_function(obs, action, next_obs)
            masks.append(mask)
        masks = torch.cat(masks,dim=0)
        
        print(f"Masks shape: {masks.shape}")
        self.ccs = [get_cc_dicts_from_mask(mask,self.num_patches,self.num_actions) for mask in masks]
        
        
    def _do_coda(self,dataset):
        def get_transition(idx):
            return *dataset[idx], self.ccs[idx]
        
        coda_obs = []
        coda_actions = []
        coda_next_obs = []
        coda_rewards = []

        self.trans_idxs = []
        if self.max_coda_transitions > 0:
            len_dataset = len(dataset)
            while len(coda_obs) < self.max_coda_transitions:
                transition_idxs = np.random.choice(len_dataset, 2, replace=False)
                transition1 = get_transition(transition_idxs[0])
                transition2 = get_transition(transition_idxs[1])
                coda_transition = do_coda_on_transitions(transition1, transition2, patch_size=self.patch_size, num_actions=self.num_actions)
                if coda_transition is not None:
                    coda_obs.extend(coda_transition[0])
                    coda_actions.extend(coda_transition[1])
                    coda_next_obs.extend(coda_transition[2])
                    self.trans_idxs.append(transition_idxs)
        else:
            if self.max_coda_transitions == -1:
                for transition_idx1 in range(len(self.dataset)):
                    for transition_idx2 in range(transition_idx1+1, len(self.dataset)):
                        transition1 = get_transition(transition_idx1)
                        transition2 = get_transition(transition_idx2)
                        coda_transition = do_coda_on_transitions(transition1, transition2, patch_size=self.patch_size[0], num_actions=self.num_actions)
                        if coda_transition is not None:
                            coda_obs.extend(coda_transition[0])
                            coda_actions.extend(coda_transition[1])
                            coda_next_obs.extend(coda_transition[2])
        
        coda_obs = torch.stack(coda_obs, dim=0)
        coda_next_obs = torch.stack(coda_next_obs, dim=0)
        coda_actions = torch.stack(coda_actions, dim=0)
        
        print("Coda observations shape: ", coda_obs.shape)
        bs_size = 100
        coda_rewards =  torch.cat([self.reward_function(coda_obs[i:i+bs_size],coda_actions[i:i+bs_size],coda_next_obs[i:i+bs_size]) for i in range(0,coda_obs.shape[0],bs_size)],dim=0)
        print("Coda rewards shape: ", coda_rewards.shape)
        coda_observations = coda_obs.cpu()
        coda_next_observations = coda_next_obs.cpu()
        coda_actions = coda_actions.cpu()
        coda_rewards = coda_rewards.cpu()
        coda_dones = torch.zeros((coda_observations.shape[0])).cpu()
        print("Coda observations shape: ", coda_observations.shape)
        return coda_observations, coda_actions, coda_next_observations, coda_rewards, coda_dones

    def save(self, path):
        np.savez(path, observations=self.observations, actions=self.actions, next_observations=self.next_observations, rewards=self.rewards, dones=self.dones)

