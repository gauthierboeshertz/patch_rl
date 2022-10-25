import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,IterableDataset
from pathlib import Path
import torch


class TransitionDataset(Dataset):
    def __init__(self, replay_dir, num_replays=None,discount=0.99) -> None:
        super(TransitionDataset).__init__()
        
        if not isinstance(replay_dir,Path):
            replay_dir = Path(replay_dir)
            
        self._replay_dir = replay_dir
        self._replays = list(replay_dir.glob('*.npz'))
        print("Found {} replays".format(len(self._replays)))
        if num_replays is not None:
            self._replays = self._replays[:num_replays]
        
        self.discount = discount
        
    def __len__(self):
        return len(self._replays) * 1000
    
    
    def __getitem__(self, idx):
        
        replay_idx = idx // 1000 # 1000 transitions per replay
        replay = np.load(self._replays[replay_idx])
        transition_idx = idx % 1000
        
        obs = (replay['observation'][transition_idx - 1]) 
        action = replay['action'][transition_idx]
        next_obs = replay['observation'][transition_idx]
        reward = replay['reward'][transition_idx]

        return torch.from_numpy(obs), torch.from_numpy(action).float(), torch.from_numpy(next_obs), torch.from_numpy(reward).float(),self.discount
        


        
class IterableTransitionDataset(IterableDataset):
    def __init__(self, replay_dir, num_replays=None) -> None:
        super(IterableTransitionDataset).__init__()
        
        self.ds = TransitionDataset(replay_dir, num_replays)            
    
    def __iter__(self):
        
        while True:
            yield self.ds[np.random.randint(0,len(self.ds))]
        
class RecursiveTransitionDataset(Dataset):
    def __init__(self, replay_dir, num_replays=None, num_steps=5) -> None:
        super(RecursiveTransitionDataset).__init__()
        
        if not isinstance(replay_dir,Path):
            replay_dir = Path(replay_dir)
            
        self._replay_dir = replay_dir
        self._replays = list(replay_dir.glob('*.npz'))
        print("Found {} replays".format(len(self._replays)))
        if num_replays is not None:
            self._replays = self._replays[:num_replays]
        
        self._num_steps = num_steps + 1 # +1 because we need to predict the next state
        self._num_idx_per_replay = 1000 // self._num_steps
    def __len__(self):
        return len(self._replays) * self._num_idx_per_replay
    
    
    def __getitem__(self, idx):
        
        replay_idx = idx // self._num_idx_per_replay 
        replay = np.load(self._replays[replay_idx])
        transition_start_idx = (idx % self._num_idx_per_replay) * self._num_steps 
        
        obss = (replay['observation'][transition_start_idx:transition_start_idx + self._num_steps])
        actions = replay['action'][transition_start_idx: transition_start_idx + self._num_steps]
        next_obss = (replay['observation'][transition_start_idx +1 : transition_start_idx + self._num_steps+1]) 

        return torch.from_numpy(obss), torch.from_numpy(actions), torch.from_numpy(next_obss)
