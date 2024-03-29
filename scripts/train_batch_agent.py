import torch
import gym
import hydra
from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from src.mask_utils import make_gt_causal_mask, get_cc_dicts_from_mask, gt_reward_function
from omegaconf import OmegaConf
import os 
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
from src.networks.cnns import Conv2dModel
import d3rlpy
import logging
from tensorboard.util import tb_logging
import wandb
from src.coda_dataset import CodaDataset
from src.datasets import ImageTransitionDataset
from src.d3rl_feature_extractor import PatchCNNFactory, PatchVAEFactory
from hydra.utils import get_original_cwd, to_absolute_path
import numpy as np
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _IgnoreTensorboardPathNotFound(logging.Filter):
    def filter(self, record):
        assert record.name == "tensorboard"
        if "No path found after" in record.msg:
            return False
        return True

def get_first_int(string):
    start = -1
    end = -1
    for i in range(len(string)):
        if string[i].isdigit():
            if start == -1:
                start = i
            end = i
        elif start != -1:
            break
    if start == -1:
        return None
    else:
        return int(string[start:end + 1])


class ImageReshaper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageReshaper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


def n_transition_list(dataset,n_transitions):
    
    transition_list = []
    if n_transitions == -1:
        n_transitions = dataset.actions.shape[0]
        
    n_obs = dataset.observations[:n_transitions].detach().numpy()
    n_actions = dataset.actions[:n_transitions].detach().numpy()
    n_next_obs = dataset.next_observations[:n_transitions].detach().numpy()
    n_rewards = dataset.rewards[:n_transitions].detach().numpy()
    n_dones = dataset.dones[:n_transitions].detach().numpy()

    if np.isnan(n_obs).any():
        print("Nan in obs")
    if np.isnan(n_actions).any():
        print("Nan in actions")
    if np.isnan(n_next_obs).any():
        print("Nan in obs")
    if np.isnan(n_rewards).any():
        print("Nan in rewards")
    if np.isnan(n_dones).any():
        print("Nan in n_dones")

    print(f"n_obs shape: {n_obs.shape}")
    print("Observations stats: ",np.min(n_obs),np.max(n_obs),np.mean(n_obs),np.std(n_obs))
    print("Rewards stats: ",np.min(n_rewards),np.max(n_rewards),np.mean(n_rewards),np.std(n_rewards))
    print("Actions stats: ",np.min(n_actions),np.max(n_actions),np.mean(n_actions),np.std(n_actions))

    terminals = n_dones * (n_rewards>0)
    for i in range(0,n_obs.shape[0]):
        trans = d3rlpy.dataset.Transition(observation_shape=list(n_obs[i].shape),
                                          action_size=n_actions[i].shape[0],
                                          observation=n_obs[i].astype(np.uint8),
                                          action=n_actions[i].astype(np.float32),
                                          reward= float(n_rewards[i]),
                                          next_observation=n_next_obs[i].astype(np.uint8),
                                          terminal= float(terminals[i]))
        transition_list.append(trans)
    
    print(f"Transition list length: {len(transition_list)}")
    return transition_list
        
    

def setup_models_for_mask(config):
    
    trained_dynamics_conf = OmegaConf.load(os.path.join(os.path.dirname(config["dynamics_model_path"]), ".hydra/config.yaml"))
    config["dynamics"] = trained_dynamics_conf["dynamics"]
    config["dynamics"]["num_actions"] = 8

    config["encoder_decoder_path"] = trained_dynamics_conf["encoder_decoder_path"]
    print("Loading encoder decoder from {}".format(config["encoder_decoder_path"]))
    encoder_decoder_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))["encoder_decoder"]
    config["encoder_decoder"] = encoder_decoder_conf
    
    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])
    config["dynamics"]["discrete_actions"] =  config["env"]["discrete_all_sprite_mover"]
    config["dynamics"]["num_rewards"] =  3
    config["encoder_decoder_path"] = trained_dynamics_conf["encoder_decoder_path"]
    
    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(device)#PatchVAE(**patch_vae).to(self.device)
    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=device))
    
    config["dynamics"]["in_features"] = config["encoder_decoder"]["embed_dim"]
    config["dynamics"]["num_patches"] = encoder_decoder.num_patches
    print(f"The encoder ouputs {encoder_decoder.num_patches} patches")
    dynamics_model = hydra.utils.instantiate(config["dynamics"]).to(device)
    dynamics_model.load_state_dict(torch.load(config["dynamics_model_path"],map_location=device))
    
    encoder_decoder.eval()
    dynamics_model.eval()
    return encoder_decoder, dynamics_model

    
def make_mask_function(image,action,next_image,  encoder=None,dynamics=None, make_gt_mask=False,patch_size=16, num_sprites=4):
    
    if make_gt_mask:
        mask = make_gt_causal_mask(image, action, next_image, num_sprites=num_sprites,patch_size=patch_size[0])
    else:
        assert encoder is not None
        mask = dynamics.get_causal_mask(image.to(device)/255., action.to(device),encoder=encoder)[0].cpu() > dynamics.causal_mask_threshold
    return mask

def make_reward_function(image, action, next_image, encoder=None,dynamics=None,make_gt_reward=False):    
    
    if make_gt_reward:
        rewards = gt_reward_function(image,next_image)
    else:
        assert encoder is not None
        rewards = dynamics.get_rewards(image, action,encoder=encoder)
    return rewards

def setup_dataset(config,env):

    if not config.dataset.use_gt_mask:
        print("Loading models for mask")
        encoder_decoder, dynamics_model = setup_models_for_mask(config)
    else:
        print("Using GT mask")
        encoder_decoder = dynamics_model = None

    mask_function = lambda image, next_image, action: make_mask_function(image, next_image,action, encoder=encoder_decoder,dynamics=dynamics_model, make_gt_mask=config.dataset.use_gt_mask,patch_size=config.dataset.patch_size, num_sprites=config.env.num_sprites)
    reward_function = lambda image, action, next_image: make_reward_function(image, action,next_image, encoder=encoder_decoder,dynamics=dynamics_model,make_gt_reward=config.dataset.use_gt_reward)

    data_path = f"{get_original_cwd()}/{config.dataset_path}"
    dataset = ImageTransitionDataset(data_path)
    print("Creating CoDA data")
    coda_dataset = CodaDataset(dataset,max_coda_transitions=config.dataset.max_coda_transitions ,mask_function=mask_function,
                                reward_function=reward_function, patch_size=config.dataset.patch_size,
                                num_actions=env.action_space._shape[0],num_patches=config.dataset.num_patches,prioritize_object_count=config.dataset.prioritize_object_count)
    if config.dataset.save_coda_dataset:
        coda_data_path = f"{get_original_cwd()}/{config.coda_save_dataset_path}"
        coda_dataset.save(coda_data_path)
        print("Saved CoDA dataset to {}".format(coda_data_path))
            
    transition_list = n_transition_list(coda_dataset,n_transitions=config.dataset.num_transitions)
    #transitiondataset_to_mdpdataset(coda_dataset)
    
    return transition_list        

def no_grad_eval(eval_env,n_trials):
    print("BRR")
    eval_func = d3rlpy.metrics.evaluate_on_environment(eval_env,n_trials=n_trials)
    def no_grad(algo,episodes):
        with torch.no_grad():
            return eval_func(algo,episodes)
    return no_grad

@hydra.main(config_path="configs", config_name="train_batch_agent")
def main(config):
    
    print("Config",config)
    set_random_seed(config["seed"])
    
    print("Evaluation environment")

    eval_env_config = bouncing_sprites.get_config(num_sprites=config.env.num_sprites,is_demo=False,
                                                            timeout_steps=config.env.episode_timesteps, 
                                                            contact_reward=True,
                                                            one_sprite_mover=config.env.one_sprite_mover,
                                                            all_sprite_mover=config.env.all_sprite_mover,
                                                            discrete_all_sprite_mover=config.env.discrete_all_sprite_mover,
                                                            random_init_places=True,
                                                            visual_obs = True,
                                                            instant_move = config.env.instant_move,
                                                            action_scale=0.05,
                                                            seed=config.seed+10,
                                                            disappear_after_contact=True,
                                                            dont_show_targets=config.env.dont_show_targets)
    
    eval_env = environment.Environment(**eval_env_config)
    eval_env = ImageReshaper(gym_wrapper.GymWrapper(eval_env))

    dataset = setup_dataset(config,eval_env)
    
    print("Replay buffer kwargs")
    if config.feature_extractor == "patch_cnn":
        encoder_factory = PatchCNNFactory(feature_dim=256,patch_size=16)
    elif config.feature_extractor == "patch_vae":
        encoder_factory = PatchVAEFactory(feature_dim=256,patch_size=16,encoder_decoder_path=config.encoder_decoder_path)
    elif config.feature_extractor == "nature_cnn":
        encoder_factory= "pixel"
    else :
        raise AssertionError("Unknown feature extractor")
    

    run = wandb.init(project=f"batch_spriteworld", entity="gboeshertz", sync_tensorboard=True,
                     config=OmegaConf.to_container(config,resolve=True),settings=wandb.Settings(start_method="thread"))
    
    wandb.run.name = f"real_{get_first_int(config.dataset_path)}_coda_{config.dataset.max_coda_transitions}_{wandb.run.name}"
    wandb.run.save()
    # prepare algorithm
    agent = d3rlpy.algos.IQL(use_gpu=torch.cuda.is_available(),
                                   actor_learning_rate=config.learning_rate,
                                   critic_learning_rate=config.learning_rate,
                                   actor_encoder_factory=encoder_factory,
                                   critic_encoder_factory=encoder_factory,
                                   batch_size=config.dataset.batch_size,
                                   scaler="pixel")
    # train
    tb_logger = tb_logging.get_logger()
    tb_logger.addFilter(_IgnoreTensorboardPathNotFound())


    agent.fit(
        dataset,
        eval_episodes=True,
        n_steps=100000,
        n_steps_per_epoch=500,
        with_timestamp=False,
        scorers={
        'eval_environment': no_grad_eval(eval_env,n_trials=15)},
        tensorboard_dir='runs',
        save_interval=500)

    run.finish()
    wandb.finish()

if __name__ == '__main__':
    main()
