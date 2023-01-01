import numpy as np
import torch
from .mask_utils import do_coda_on_transitions,get_cc_dicts_from_mask
import time
import einops 

from IPython import embed


class CodaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mask_function, reward_function, max_coda_transitions=-1,
                 patch_size=(16,16), num_actions=4,num_patches=64):
        
        self.mask_function = mask_function
        self.reward_function = reward_function
        self.max_coda_transitions = max_coda_transitions
        self.patch_size = patch_size
        self.num_actions = num_actions
        self.num_patches = num_patches
        
        print("Dataset obs max", dataset.observations.float().max())
        print("Dataset obs min", dataset.observations.float().min())
        print("Dataset obs mean", dataset.observations.float().float().mean())
        print("Dataset obs std", dataset.observations.float().float().std())

        self._create_all_ccs(dataset)
        s_time = time.time()
        coda_observations, coda_actions, coda_next_observations, coda_rewards, coda_dones = self._do_coda(dataset)
        print("Time to do coda", time.time()-s_time)


        if self.max_coda_transitions == 0:
            self.actions = dataset.actions
            self.next_observations = dataset.next_observations.byte()
            self.rewards = dataset.rewards
            self.dones = dataset.dones
            self.observations = torch.cat((dataset.observations, torch.zeros_like(coda_observations)), dim=0).byte()[:self.dones.shape[0]]# Without this line, the training doesnt work
            
        else:

            obss = dataset.observations
            rewss  = dataset.rewards
            actss = dataset.actions
            nobss = dataset.next_observations
            print("Observations stats: ",obss.dtype,torch.min(obss),torch.max(obss),torch.mean(obss.float()),torch.std(obss.float()))
            print("Rewards stats: ", rewss.dtype,torch.min(rewss),torch.max(rewss),torch.mean(rewss),torch.std(rewss))
            print("Actions stats: ", actss.dtype ,torch.min(actss),torch.max(actss),torch.mean(actss),torch.std(actss))
            print("Next observations stats: ",nobss.dtype,torch.min(nobss),torch.max(nobss),torch.mean(nobss.float()),torch.std(nobss.float()))
            
            
            obss = coda_observations
            rewss  = coda_rewards
            actss = coda_actions
            nobss = coda_next_observations
            print("Observations stats: ",obss.dtype,torch.min(obss),torch.max(obss),torch.mean(obss.float()),torch.std(obss.float()))
            print("Rewards stats: ", rewss.dtype,torch.min(rewss),torch.max(rewss),torch.mean(rewss),torch.std(rewss))
            print("Actions stats: ", actss.dtype ,torch.min(actss),torch.max(actss),torch.mean(actss),torch.std(actss))
            print("Next observations stats: ",nobss.dtype,torch.min(nobss),torch.max(nobss),torch.mean(nobss.float()),torch.std(nobss.float()))

            self.observations = torch.cat((dataset.observations, coda_observations), dim=0).type(torch.uint8)
            self.actions = torch.cat((dataset.actions, coda_actions), dim=0)
            self.next_observations = torch.cat((dataset.next_observations, coda_next_observations), dim=0).type(torch.uint8)
            self.rewards = torch.cat((dataset.rewards, coda_rewards), dim=0)
            self.dones = torch.cat((dataset.dones, coda_dones), dim=0)
            obss = self.observations
            rewss  = self.rewards
            actss = self.actions
            nobss = self.next_observations
            print("Observations stats: ",obss.dtype,torch.min(obss),torch.max(obss),torch.mean(obss.float()),torch.std(obss.float()))
            print("Rewards stats: ", rewss.dtype,torch.min(rewss),torch.max(rewss),torch.mean(rewss),torch.std(rewss))
            print("Actions stats: ", actss.dtype ,torch.min(actss),torch.max(actss),torch.mean(actss),torch.std(actss))
            print("Next observations stats: ",nobss.dtype,torch.min(nobss),torch.max(nobss),torch.mean(nobss.float()),torch.std(nobss.float()))

        
    
    def _create_all_ccs(self,dataset):
        masks = []
        batch_size = 100
        for i in range(0,len(dataset),batch_size):
            max_idx = min(i+batch_size,len(dataset))
            obs = dataset.observations[i:max_idx]
            action = dataset.actions[i:max_idx]
            next_obs = dataset.next_observations[i:max_idx]
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
        if self.max_coda_transitions >= 0:
            len_dataset = len(dataset)
            while len(coda_obs) < self.max_coda_transitions+1:
                transition_idxs = np.random.choice(len_dataset, 1, replace=False)
                transition_idxs = np.append(transition_idxs, np.random.choice(list(range(0,max(transition_idxs[0]-20,0))) + list(range(min(transition_idxs[0]+20,len_dataset),len_dataset)), 1))
                transition1 = get_transition(transition_idxs[0])
                transition2 = get_transition(transition_idxs[1])
                coda_transition = do_coda_on_transitions(transition1, transition2, patch_size=self.patch_size, num_actions=self.num_actions)
                if coda_transition is not None:
                    coda_obs.extend(coda_transition[0])
                    coda_actions.extend(coda_transition[1])
                    coda_next_obs.extend(coda_transition[2])
                    self.trans_idxs.append(transition_idxs)
        else:
            for transition_idx1 in range(len(dataset)):
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

        return coda_observations, coda_actions, coda_next_observations, coda_rewards, coda_dones

    def save(self, path):
        np.savez(path, states=self.observations, actions=self.actions, next_states=self.next_observations, rewards=self.rewards, dones=self.dones)

