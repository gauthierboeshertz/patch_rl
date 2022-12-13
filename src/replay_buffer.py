import numpy as np
import torch
from .patch_utils import image_to_patches
import math
from .mask_utils import get_ccs_to_swap, swap_transition_components
import random

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer():
    def __init__(self, obs_dim, act_dim, size):
        
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,1), dtype=np.float32)
        self.done_buf = np.zeros((size,1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        

    def sample_batch(self, batch_size=32,):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs = self.obs_buf[idxs]
        batch_obs2 = self.obs2_buf[idxs]
        batch_act = self.act_buf[idxs]
        batch_rew = self.rew_buf[idxs]
        batch_done = self.done_buf[idxs]
        
        
        
        batch = dict(obs=batch_obs,
                next_obs=batch_obs2,
                action=batch_act,
                reward=batch_rew,
                done=batch_done)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def store(self, obs, act, rew, next_obs, done):
        
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)



        

class CodaPatchReplayBuffer():
    def __init__(self, ccs_function, num_patches,patch_size, obs_dim, act_dim, size,coda_size):
        
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.coda_size = coda_size
        self.ccs_function = ccs_function
        self.patch_size = patch_size
        self.num_patches = num_patches
        

    def sample_batch(self, batch_size=32,coda_batch_size=0):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs = self.obs_buf[idxs]
        batch_obs2 = self.obs2_buf[idxs]
        batch_act = self.act_buf[idxs]
        batch_rew = self.rew_buf[idxs]
        batch_done = self.done_buf[idxs]
        
        
        if coda_batch_size > 0:
            coda_idxs = np.random.randint(0, self.coda_size, size=coda_batch_size)
            coda_batch_obs = self.coda_obs_buf[coda_idxs]
            coda_batch_obs2 = self.coda_obs2_buf[coda_idxs]
            coda_batch_act = self.coda_act_buf[coda_idxs]
            coda_batch_rew = self.coda_rew_buf[coda_idxs]
            coda_batch_done = self.coda_done_buf[coda_idxs]
            
            batch_obs = np.concatenate((batch_obs,coda_batch_obs),axis=0)
            batch_obs2 = np.concatenate((batch_obs2,coda_batch_obs2),axis=0)
            batch_act = np.concatenate((batch_act,coda_batch_act),axis=0)
            batch_rew = np.concatenate((batch_rew,coda_batch_rew),axis=0)
            batch_done = np.concatenate((batch_done,coda_batch_done),axis=0)
            
        batch = dict(obs=batch_obs,
                obs2=batch_obs2,
                act=batch_act,
                rew=batch_rew,
                done=batch_done)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def store(self, obs, act, rew, next_obs, done):
        
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_transition_data(self, idx):
        image = self.obs_buf[idx]
        action = self.act_buf[idx]
        next_image = self.obs2_buf[idx]
        reward = self.rew_buf[idx]
        done = self.done_buf[idx]
        ccs = self.ccs_function(image)
        return image, action, next_image, reward, done, ccs
    
    def can_do_coda(self,reward1, reward2, done1, done2):
        return reward1 == 0 and reward2 == 0 and done1 == 0 and done2 == 0
    
    def do_coda_on_transition(self, t1_idx, t2_idx):
        
        image1, action1, next_image1, reward1, done1, ccs1 = self.get_transition_data(t1_idx)    
        image2, action2, next_image2, reward2, done2, ccs2 = self.get_transition_data(t2_idx)
        
        if self.can_do_coda(reward1, reward2, done1, done2):
            print("Cant do coda on transitions with rewards")
            
        patches1 = image_to_patches(image1, patch_size=self.patch_size,num_patches_sqrt=image1.shape[2]//self.patch_size[0])
        patches2 = image_to_patches(image2, patch_size=self.patch_size,num_patches_sqrt=image2.shape[2]//self.patch_size[0])

        next_patches1 = image_to_patches(next_image1, patch_size=self.patch_size,num_patches_sqrt=image1.shape[2]//self.patch_size[0])
        next_patches2 = image_to_patches(next_image2, patch_size=self.patch_size,num_patches_sqrt=image2.shape[2]//self.patch_size[0])

        num_patches = patches1.shape[0]
        comp_to_swap_ccs1, comp_to_swap_ccs2 = get_ccs_to_swap(ccs1, ccs2, num_patches, self.act_dim)
        
        coda_transitions = swap_transition_components(patches1,patches2, action1,action2, next_patches1, next_patches2, 
                                            comp_to_swap_ccs1,comp_to_swap_ccs2)
        
        
        return coda_transitions


    def process_coda_transitions(self, coda_transitions):
        self.coda_obs_buf = np.zeros(combined_shape(self.coda_size, self.obs_dim), dtype=np.float32)
        self.coda_obs2_buf = np.zeros(combined_shape(self.coda_size, self.obs_dim), dtype=np.float32)
        self.coda_act_buf = np.zeros(combined_shape(self.coda_size, self.act_dim), dtype=np.float32)
        self.coda_rew_buf = np.zeros(self.coda_size, dtype=np.float32)
        self.coda_done_buf = np.zeros(self.coda_size, dtype=np.float32)

        coda_obs = []
        coda_obs2 = []
        coda_act = []
        for coda_transition in coda_transitions:
            coda_obs = coda_obs + coda_transition[0]
            coda_act = coda_act + coda_transition[1]
            coda_obs2 = coda_obs2 + coda_transition[2]
        
        self.coda_obs = np.array(coda_obs)[np.random.randint(0, self.size, size=self.coda_size)]
        self.coda_obs2 = np.array(coda_obs2)[np.random.randint(0, self.size, size=self.coda_size)]
        self.coda_act = np.array(coda_act)[np.random.randint(0, self.size, size=self.coda_size)]
        self.coda_rew = np.zeros(self.coda_size, dtype=np.float32)
        self.coda_done = np.zeros(self.coda_size, dtype=np.float32)
        
        print("Coda obs shape: ", self.coda_obs.shape)
        
    def do_coda_all(self):
        
        coda_transitions = []
        max_idx = self.ccs_buf.index(-1)
        for t1_idx in range(max_idx):
            for t2_idx in range(t1_idx+1, max_idx):
                coda_transitions.append(self.do_coda_on_transition(self, t1_idx, t2_idx))
        
        self.coda_transitions = coda_transitions