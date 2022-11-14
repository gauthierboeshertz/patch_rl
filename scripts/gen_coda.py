import numpy as np
from functools import reduce
from itertools import combinations, chain
from scipy.sparse.csgraph import connected_components
import random
import torch
from src.mask_utils import make_gt_causal_mask
from src.datasets import SequenceImageTransitionDataset

def get_default_mask(s, a):
    """ assumes set-based s and a... so shape should be (n_componenents, *component_shape) """
    if len(a.shape) == 1:
        a.reshape(1, -1)

    mask_dim = len(s) + len(a)
    mask_shape = (mask_dim, mask_dim)
    return np.ones(mask_shape)


def batch_get_default_mask(s, a):
    """ Batch version of get default mask """
    s_shape = s.shape
    a_shape = a.shape
    if len(a_shape) == 2:
        assert len(a_shape) > 1
        a.reshape(-1, 1, a_shape[-1])

    mask_dim = s_shape[1] + a_shape[1]
    mask_shape = (s_shape[0], mask_dim, mask_dim)
    return np.ones(mask_shape)


def get_cc_from_mask(mask):
    """
    Converts a mask into a list of CC indices tuples.
    E.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
    this will return [array([0]), array([1]), array([2, 3])]

    Note that the mask should be a square, so in case we have (s, a) x (s2,),
    we should first dummy a2 columns to form a square mask. 
    """
    ccs = connected_components(mask)
    num_ccs, cc_idxs = ccs
    return [np.where(cc_idxs == i)[0] for i in range(num_ccs)]


def powerset(n):
    xs = list(range(n))
    return list(chain.from_iterable(combinations(xs, n) for n in range(n + 1)))


def reduce_cc_list_by_union(cc_list, max_ccs):
    """Takes a cc list that is too long and merges some components to bring it
    to max_ccs"""
    while len(cc_list) > max_ccs:
        i, j = np.random.choice(range(1, len(cc_list) - 1), 2, replace=False)
        if (j == 0) or (j == len(cc_list) - 1):
            continue  # don't want to delete the base
        cc_list[i] = np.union1d(cc_list[i], cc_list[j])
        del cc_list[j]
    return cc_list


def disentangled_components(cc_lst):
    """Converts connected component list into a list of disentangled subsets
    of the indices.
    """
    subsets = powerset(len(cc_lst))
    res = []
    for subset in subsets:
        res.append(reduce(np.union1d, [cc_lst[i] for i in subset], np.array([])).astype(np.int64))
    return set(map(tuple, res))


def get_dcs_from_mask(mask, max_ccs = 6):
    cc = get_cc_from_mask(mask)
    return disentangled_components(reduce_cc_list_by_union(cc, max_ccs))


def make_gt_trans_coda(dataset, t_idx1, t_idx2):
    
    image1, action1 = dataset[t_idx1][0], dataset[t_idx1][1]
    image2, action2 = dataset[t_idx2][0], dataset[t_idx2][1]
    mask1 = make_gt_causal_mask(image1.unsqueeze(0),action1[0],patch_size=16,num_sprites=4)
    mask2 = make_gt_causal_mask(image2.unsqueeze(0),action2[0],patch_size=16,num_sprites=4)

    m1 = mask1.cpu().numpy().transpose(0,2,1)[0]
    m2 = mask2.cpu().numpy().transpose(0,2,1)[0]

    t1_sa = np.arange(m1.shape[0])
    t1_s2 = np.arange(m1.shape[0] - action1.shape[1])

    t2_sa = np.arange(m2.shape[0])
    t2_s2 = np.arange(m2.shape[0] - action2.shape[1])

    t1 = (t1_sa,t1_s2)
    t2 = (t2_sa,t2_s2)
    return (t1,m1), (t2,m2)


def main():

    data_path = "/cluster/home/gboeshertz/patch_rl/data/visual_150transitions_4_discrete_all_sprite_mover_True_4.npz"
    #data_path = "/cluster/home/gboeshertz/patch_rl/data/visual_180transitions_4_all_sprite_mover_False_4instantmove.npz"
    print(data_path)
    dataset = SequenceImageTransitionDataset(data_path=data_path)

    t_idx1 = 10
    t_idx2 = 100
    t1m1, t2m2 = make_gt_trans_coda(dataset, t_idx1, t_idx2)
    
        
    sa1, s21 = t1m1[0]
    sa2, s22 = t2m2[0]
    m1 = t1m1[1]
    m2 = t2m2[1]
    max_ccs = 30

    print("creating disconnected components")
    dc1 = get_dcs_from_mask(m1, max_ccs)
    dc2 = get_dcs_from_mask(m2, max_ccs)
    
    print("Created {} disconnected components for t1".format(len(dc1)))
    print("Created {} disconnected components for t2".format(len(dc2)))
    
    max_samples = 100

    # get shared connected components in random order
    shared_dc = list(dc1.intersection(dc2))
    random.shuffle(shared_dc)

    print("shared disconnected components")
    print("len shared_dc", len(shared_dc))
    # subsample shared_dc down to max_samples
    if len(shared_dc) > max_samples:
        shared_dc = shared_dc[:max_samples]

    
    all_idxs = set(range(len(sa1)))
    res = []
    for dc in shared_dc:
        not_dc = list(all_idxs - set(dc))
        dc = list(dc) # (0, 2)

        print(len(dc))
        proposed_sa = np.zeros_like(sa1)
        proposed_s2 = np.zeros_like(sa1)

        proposed_sa[dc]     = sa1[dc]
        proposed_sa[not_dc] = sa2[not_dc]
        proposed_s2[dc]     = s21[dc]
        proposed_s2[not_dc] = s22[not_dc]

        proposed_t = (proposed_sa, proposed_s2)
        res.append(proposed_t)
    return res

    
if __name__ == "__main__":

    main()
    