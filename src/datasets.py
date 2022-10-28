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
        self.observations = torch.from_numpy(data["states"])
        #self.observations = torch.cat((self.observations[:,0],self.observations[:,1],self.observations[:,2]),dim=-1).permute(0,3,1,2)
        self.observations = rearrange(self.observations,'n b h w c  -> n (b c) h w')
        self.next_observations = torch.from_numpy(data["next_states"])
        self.next_observations = rearrange(self.next_observations,'n b h w c  -> n (b c) h w')
        self.actions = torch.from_numpy(data["actions"])
        self.rewards = torch.from_numpy(data["rewards"])
        self.dones = torch.from_numpy(data["dones"])

    def __len__(self):
        return self.observations.shape[0]
    
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.next_observations[idx], self.rewards[idx]


class SequenceImageTransitionDataset(Dataset):
    
    def __init__(self, data_path:str,sequence_length=5) -> None:
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
        data = np.load(data_path)
        self.observations = torch.from_numpy(data["states"])
        #self.observations = torch.cat((self.observations[:,0],self.observations[:,1],self.observations[:,2]),dim=-1).permute(0,3,1,2)
        self.observations = rearrange(self.observations,'n b h w c  -> n (b c) h w')
        self.next_observations = torch.from_numpy(data["next_states"])
        self.next_observations = rearrange(self.next_observations,'n b h w c  -> n (b c) h w')
        self.actions = torch.from_numpy(data["actions"])
        self.rewards = torch.from_numpy(data["rewards"])
        self.dones = torch.from_numpy(data["dones"])
        
        self.valid_transitions_indices = self.get_valid_indices()

    def get_valid_indices(self):
        """
        make_episodes  make episode by adding indices to episode list using dones,
                        also removes the last sequence_length transitions of each episode
    
        _extended_summary_

        Returns:
            _description_
        """        
        episodes = []
        episode = []
        for i in range(len(self.dones)):
            episode.append(i)
            if self.dones[i]:
                episodes.append(episode)
                episode = []
        
        cut_episodes = []
        for ep in episodes:
            cut_episodes.append(ep[:-self.sequence_length])
        return [item for sublist in cut_episodes for item in sublist] #flatten list to get indices

    def __len__(self):
        return  len(self.valid_transitions_indices)
            
            
    def __getitem__(self, idx):
        
        transition_idx = self.valid_transitions_indices[idx]
        
        obs = []
        actions = []
        next_obs = []
        rewards = []
        
        for i in range(self.sequence_length):
            obs.append(self.observations[transition_idx+i])
            actions.append(self.actions[transition_idx+i])
            next_obs.append(self.next_observations[transition_idx+i])
            rewards.append(self.rewards[transition_idx+i])
        
        return torch.stack(obs).float(), torch.stack(actions).float(), torch.stack(next_obs).float(), torch.stack(rewards).float()

