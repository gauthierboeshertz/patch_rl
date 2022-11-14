import argparse
import numpy as np
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import os 

def dataset_metrics(dataset_path):

    data = np.load(os.path.join(get_original_cwd(),f"{dataset_path}.npz"))
    print("Dataset states shape",data["states"].shape)
    print("Dataset actions shape",data["actions"].shape)
    print("Action statistics")
    
    num_sprites= data["actions"].shape[1]
    num_actions = np.unique(data["actions"]).shape[0]
    
    for action in range(num_actions):
        for sprite in range(num_sprites):
            print(f"For action {action} and sprite {sprite} the percentage is {np.sum(data['actions'][:,sprite] == action)/data['actions'].shape[0]}")
    
    
@hydra.main(config_path="configs", config_name="dataset_metrics")
def main(config):
    data_name = "../data/{}_{}transitions_{}_{}_{}_{}{}".format("visual" if config["visual_obs"] else "states",config["num_transitions"],config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "discrete_all_sprite_mover" if config["discrete_all_sprite_mover"] else "select_move"),config["random_init_places"],config["num_action_repeat"],"instantmove" if config["instant_move"] else "")
    print("Data name",data_name)
    dataset_metrics(data_name)
    

if __name__ == "__main__":

    main()


