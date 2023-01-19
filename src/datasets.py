import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,IterableDataset
from pathlib import Path
import torch
from einops import rearrange

class ImageTransitionDataset(Dataset):
    def __init__(self, data_path) -> None:
        """
        __init__ load a transition dataset, getting an item results in getting one transition,
                    i.e a tuple of (obs, action, next_obs, reward,done)

        _extended_summary_

        Args:
            data_path (str): path of a npz file containing the dataset
        """        
        super(ImageTransitionDataset).__init__()
        assert data_path is not None 
        data = np.load(data_path)
        self.observations = torch.from_numpy(data["states"]).type(torch.uint8)
        #self.observations = torch.cat((self.observations[:,0],self.observations[:,1],self.observations[:,2]),dim=-1).permute(0,3,1,2)
        if not self.observations.shape == (5012, 3, 128, 128):
            self.observations = rearrange(self.observations,'n b h w c  -> n (b c) h w').type(torch.uint8)
        print("Observations shape",self.observations.shape)
        self.next_observations = torch.from_numpy(data["next_states"]).type(torch.uint8)
        if not self.observations.shape == (5012, 3, 128, 128):
            self.next_observations = rearrange(self.next_observations,'n b h w c  -> n (b c) h w').type(torch.uint8)
        self.actions = torch.from_numpy(data["actions"]).float()
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()

    def __len__(self):
        return self.observations.shape[0]
    
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.next_observations[idx], self.rewards[idx], self.dones[idx]



class TransitionMaskDataset(Dataset):
    def __init__(self, observations,actions,next_observations, mask_function,rewards=None) -> None:
        """
        __init__ load a transition dataset, getting an item results in getting one transition,
                    i.e a tuple of (obs, action, next_obs, reward,done)

        _extended_summary_

        Args:
            data_path (str): path of a npz file containing the dataset
        """        
        super(ImageTransitionDataset).__init__()
        self.obs = observations
        self.actions = actions
        self.next_obs = next_observations
        self.mask_function = mask_function
        self.rewards = rewards


    def _create_mask(self):
        masks = []
        batch_size = 100
        for batch_idx in range(0,self.obs.shape[0],batch_size):
            max_idx = min(batch_idx+batch_size,self.obs.shape[0])
            masks.append(self.mask_function(self.obs[batch_idx:max_idx],self.actions[batch_idx:max_idx],self.next_obs[batch_idx:max_idx]))
        
        
    def __len__(self):
        return self.observations.shape[0]
    
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.next_observations[idx], self.rewards[idx], self.dones[idx]


class SequenceImageTransitionDataset(Dataset):
    
    def __init__(self, data_path:str,sequence_length=5, onehot_action=False,is_test=False) -> None:
        """
        __init__ loads a transition dataset, getting an item results in getting sequence_length transition
                

        _extended_summary_

        Args:
            data_path (str): path of a npz file containing the dataset
            sequence_length (int, optional): _description_. Defaults to 5.
        """            
        super(SequenceImageTransitionDataset).__init__()
        assert data_path is not None 
        self.sequence_length = sequence_length
        self.is_test = is_test
        data = np.load(data_path)
        self.observations = torch.from_numpy(data["states"])
        print(self.observations.shape)
        #self.observations = torch.cat((self.observations[:,0],self.observations[:,1],self.observations[:,2]),dim=-1).permute(0,3,1,2)
        self.observations = rearrange(self.observations,'n b h w c  -> n (b c) h w')
        self.next_observations = torch.from_numpy(data["next_states"])
        self.next_observations = rearrange(self.next_observations,'n b h w c  -> n (b c) h w')
        self.actions = torch.from_numpy(data["actions"])
        self.rewards = torch.from_numpy(data["rewards"])
        self.dones = torch.from_numpy(data["dones"])
        if is_test:
            self.infos = torch.from_numpy(data["infos"])
            print(self.infos)
            print(self.infos.shape)
        
        self.valid_transitions_indices = self._get_valid_indices()
        
        #if  self.is_test:
        #    self.observations, self.actions, self.next_observations, self.rewards, self.infos = self._make_all_sequence() 
        #else:
        #    self.observations, self.actions, self.next_observations, self.rewards = self._make_all_sequence() 
        #assert self.observations.shape[0] == self.actions.shape[0] == self.next_observations.shape[0] == self.rewards.shape[0] == len(self.valid_transitions_indices)
        if onehot_action:
            self.actions = nn.functional.one_hot(self.actions.long(),num_classes=4).float().permute(0,1,3,2)
        self._classify_rewards()
        
    def _classify_rewards(self):
        """
        _classify_rewards 

        _extended_summary_
        """
        all_rewards = torch.unique(self.rewards)
        self.num_rewards = len(all_rewards)
        for reward_idx, reward in enumerate(all_rewards):
            print(f"reward {reward} has {torch.sum(self.rewards == reward)} transitions")
            self.rewards[self.rewards == reward] = reward_idx
        
    def _get_valid_indices(self):
        """
        make_episodes  make episode by adding indices to episode list using dones,
                        also removes the last sequence_length transitions of each episode
    
        _extended_summary_

        Returns:
            _description_
        """        
        episodes = []
        episode = []
        for i in range(self.dones.shape[0]):
            episode.append(i)
            if self.dones[i]:
                episodes.append(episode)
                episode = []
        episodes.append(episode)
        cut_episodes = []
        for ep in episodes:
            cut_episodes.append(ep[:-self.sequence_length])
            
        return [item for sublist in cut_episodes for item in sublist] #flatten list to get indices

    def __len__(self):
        return  len(self.valid_transitions_indices)
            
    def _make_all_sequence(self):
        """
        _make_sequence  Creates all the sequence of size sequence_length starting from idx

        _extended_summary_


        Returns:
            _description_ : tuple of (obs, actions, next_obs, rewards)
        """        
        all_obs = []
        all_actions = []
        all_next_obs = []
        all_rewards = []
        all_infos = []
        for t_idx in self.valid_transitions_indices:
            obs = []
            actions = []
            next_obs = []
            rewards = []
            infos = []
            for i in range(self.sequence_length):
                obs.append(self.observations[t_idx+i])
                actions.append(self.actions[t_idx+i])
                next_obs.append(self.next_observations[t_idx+i])
                rewards.append(self.rewards[t_idx+i])
                if self.is_test:
                    infos.append(self.infos[t_idx+i])
                    
            all_obs.append(torch.stack(obs).float())
            all_actions.append(torch.stack(actions).float())
            all_next_obs.append(torch.stack(next_obs).float())
            all_rewards.append(torch.stack(rewards).float())
            if self.is_test:
                all_infos.append(torch.stack(infos).float())
        if self.is_test:
            return torch.stack(all_obs), torch.stack(all_actions), torch.stack(all_next_obs), torch.stack(all_rewards), torch.stack(all_infos)
        else:
            return torch.stack(all_obs), torch.stack(all_actions), torch.stack(all_next_obs), torch.stack(all_rewards)#torch.stack(obs).float(), torch.stack(actions).float(), torch.stack(next_obs).float(), torch.stack(rewards).float()

        
    def __getitem__(self, idx):

        t_idx = self.valid_transitions_indices[idx]
        obs = []
        actions = []
        next_obs = []
        rewards = []
        for i in range(self.sequence_length):
            obs.append(self.observations[t_idx+i])
            actions.append(self.actions[t_idx+i])
            next_obs.append(self.next_observations[t_idx+i])
            rewards.append(self.rewards[t_idx+i])
        
        obs = torch.stack(obs).float()
        actions = torch.stack(actions).float()
        next_obs = torch.stack(next_obs).float()
        rewards = torch.stack(rewards).float()
        return obs, actions, next_obs, rewards

