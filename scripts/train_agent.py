from copy import deepcopy
import numpy as np
import torch
import gym
import hydra
from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from stable_baselines3 import TD3
import stable_baselines3
import numpy as np
from src.sb_replay_buffer import CodaPatchReplayBuffer
from src.mask_utils import make_gt_causal_mask, get_cc_dicts_from_mask, gt_reward_function
from omegaconf import OmegaConf
import os 
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
from typing import Callable
from src.sb_patch_feature_extractor import PatchVAEFeatureExtractor,PatchFeatureExtractor
from src.networks.cnns import Conv2dModel
from stable_baselines3.common.noise import NormalActionNoise

import logging

from tensorboard.util import tb_logging
import wandb


class _IgnoreTensorboardPathNotFound(logging.Filter):
    def filter(self, record):
        assert record.name == "tensorboard"
        if "No path found after" in record.msg:
            return False
        return True



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def setup_models_for_mask(config,num_actions=4):
    
    trained_dynamics_conf = OmegaConf.load(os.path.join(os.path.dirname(config["dynamics_model_path"]), ".hydra/config.yaml"))

    config["encoder_decoder_path"] = trained_dynamics_conf["encoder_decoder_path"]
    print("Loading encoder decoder from {}".format(config["encoder_decoder_path"]))
    encoder_decoder_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))["encoder_decoder"]
    config["encoder_decoder"] = encoder_decoder_conf

    config["encoder_decoder"]["in_channels"] = (3 if (config["env"]["instant_move"] or config["env"]["discrete_all_sprite_mover"]) else 9)
    print(config["encoder_decoder"]["in_channels"])
    config["dynamics"]["discrete_actions"] =  True

    encoder_decoder = hydra.utils.instantiate(config["encoder_decoder"]).to(config["device"])#PatchVAE(**patch_vae).to(self.device)
    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=config["device"]))
    
    config["dynamics"] = trained_dynamics_conf["dynamics"]
    config["dynamics"]["in_features"] = config["encoder_decoder"]["embed_dim"]
    config["dynamics"]["num_patches"] = encoder_decoder.num_patches
    config["dynamics"]["num_actions"] = num_actions

    print(f"The encoder ouputs {encoder_decoder.num_patches} patches")
    dynamics_model = hydra.utils.instantiate(config["dynamics"]).to(config["device"])
    dynamics_model.load_state_dict(torch.load(config["dynamics_model_path"],map_location=config["device"]))
    
    encoder_decoder.eval()
    dynamics_model.eval()
    return encoder_decoder, dynamics_model
    
def make_mask_function(image, action,  next_image, encoder=None,dynamics=None, make_gt_mask=False,patch_size=16, num_sprites=4):
    
    if make_gt_mask:
        mask = make_gt_causal_mask(image, action, next_image,num_sprites=num_sprites,patch_size=patch_size)
    else:
        assert encoder is not None
        mask = dynamics.get_causal_mask(image, action,encoder=encoder, discard_ratio=0.95,head_fusion=dynamics.head_fusion)
    return mask

def make_reward_function(image,  action,  next_image,encoder=None,dynamics=None,make_gt_reward=False):    
    
    if make_gt_reward:
        rewards = gt_reward_function(image,next_image)
    else:
        assert encoder is not None
        rewards = dynamics.get_rewards(image, action,encoder=encoder)
    return rewards


def setup_custom_vae_feature_extractor(config):
    trained_vae_conf = OmegaConf.load(os.path.join(os.path.dirname(config["encoder_decoder_path"]), ".hydra/config.yaml"))
    encoder_decoder = hydra.utils.instantiate(trained_vae_conf["encoder_decoder"]).to(config["device"])

    encoder_decoder.load_state_dict(torch.load(config["encoder_decoder_path"],map_location=config["device"]))
    for param in encoder_decoder.parameters():
        param.requires_grad = False
    encoder_decoder.eval()
    
    feat_extractor_kwargs = {"encoder_decoder":encoder_decoder,"features_dim":512}
    return feat_extractor_kwargs

@hydra.main(config_path="configs", config_name="train_agent")
def main(config):
    
    set_random_seed(config["seed"])
    

    print("Training environment")
    env_config = bouncing_sprites.get_config(num_sprites=config.env.num_sprites,is_demo=False,timeout_steps=config.env.episode_timesteps, 
                                                            contact_reward=True,
                                                            one_sprite_mover=config.env.one_sprite_mover,
                                                            all_sprite_mover=config.env.all_sprite_mover,
                                                            discrete_all_sprite_mover=config.env.discrete_all_sprite_mover,
                                                            random_init_places=config.env.random_init_places,
                                                            visual_obs = True,
                                                            instant_move = config.env.instant_move,
                                                            action_scale=0.05,
                                                            seed=config.seed,
                                                            disappear_after_contact=True,
                                                            dont_show_targets=config.env.dont_show_targets)
    

    env = environment.Environment(**env_config)
    env = gym_wrapper.GymWrapper(env)
    
    print("Evaluation environment")
    eval_env_config = bouncing_sprites.get_config(num_sprites=config.env.num_sprites,is_demo=False,timeout_steps=config.env.episode_timesteps, 
                                                            contact_reward=True,
                                                            one_sprite_mover=config.env.one_sprite_mover,
                                                            all_sprite_mover=config.env.all_sprite_mover,
                                                            discrete_all_sprite_mover=config.env.discrete_all_sprite_mover,
                                                            random_init_places=config.env.random_init_places,
                                                            visual_obs = True,
                                                            instant_move = config.env.instant_move,
                                                            action_scale=0.05,
                                                            seed=config.seed,
                                                            disappear_after_contact=True,
                                                            dont_show_targets=config.env.dont_show_targets)
    
    eval_env = environment.Environment(**eval_env_config)
    eval_env = gym_wrapper.GymWrapper(eval_env)

    print("Setting up models")
    if config.replay_buffer.coda_minimum_size > 0:
        encoder_decoder, dynamics_model = None,None
        if not config.replay_buffer.use_gt_mask:
            encoder_decoder, dynamics_model = setup_models_for_mask(config,num_actions=env.action_space._shape[0])
        mask_function = lambda image,  action, next_image,: make_mask_function(image, action, next_image, encoder=encoder_decoder,dynamics=dynamics_model, make_gt_mask=config.replay_buffer.use_gt_mask,patch_size=config.encoder_decoder.patch_size, num_sprites=config.env.num_sprites)
        reward_function = lambda image, action, next_image: make_reward_function(image, action, next_image, encoder=encoder_decoder,dynamics=dynamics_model,make_gt_reward=config.replay_buffer.use_gt_reward)
    else:
        mask_function = None
        reward_function = None
        
    replay_buffer_kwargs = OmegaConf.to_container(config.replay_buffer,resolve=True)
    del replay_buffer_kwargs["use_gt_mask"]
    del replay_buffer_kwargs["use_gt_reward"]
    replay_buffer_kwargs["mask_function"] = mask_function
    replay_buffer_kwargs["reward_function"] = reward_function

    #replay_buffer_kwargs["action_space"] = env.action_space
    #replay_buffer_kwargs["observation_space"] = env.observation_space
    
    print("Replay buffer kwargs")
    print(replay_buffer_kwargs)
    if config.feature_extractor == "patch_vae":
        assert config.encoder_decoder_path != "", "Must provide path to trained VAE"
        print("Setting up PatchVAE custom feature extractor")
        custom_feature_extractor_class = PatchVAEFeatureExtractor
        custom_feature_extractor_kwargs = setup_custom_vae_feature_extractor(config)
    elif config.feature_extractor == "patch_cnn":
        #cnn_dowsample = Conv2dModel(in_channels=3, channels=[64,128,64,32],kernel_sizes=[5,3,3,3],strides=[2,2,2,2],paddings=[1,1,1,1],norm_type="gn")
        custom_feature_extractor_kwargs = {"features_dim":256, "patch_size":16}
        custom_feature_extractor_class = PatchFeatureExtractor
    elif config.feature_extractor == "nature_cnn":
        custom_feature_extractor_class= stable_baselines3.common.torch_layers.NatureCNN
        custom_feature_extractor_kwargs = None
    else :
        raise AssertionError("Unknown feature extractor")
    
    run = wandb.init(project=f"td3_spriteworld_{config.env.num_sprites}_sprites", entity="gboeshertz", sync_tensorboard=True,
                     config=OmegaConf.to_container(config,resolve=True),settings=wandb.Settings(start_method="thread"))
    
    wandb.run.name = f"{config.feature_extractor}_{wandb.run.name}"
    wandb.run.save()

    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("CnnPolicy", env, verbose=1,seed=config.seed,device="auto",learning_rate=linear_schedule(0.0001),
                replay_buffer_class=CodaPatchReplayBuffer,replay_buffer_kwargs=replay_buffer_kwargs,
                buffer_size=100000, action_noise=action_noise,
                    policy_kwargs={"features_extractor_class":custom_feature_extractor_class,"features_extractor_kwargs":custom_feature_extractor_kwargs})

    tb_logger = tb_logging.get_logger()
    tb_logger.addFilter(_IgnoreTensorboardPathNotFound())

    logger = configure("runs/", ["stdout", "tensorboard"])
    model.set_logger(logger)

    wandbcallback = WandbCallback(
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=15,
        best_model_save_path="./eval_runs/",
        log_path="eval_runs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=config.learn_timesteps,callback=[eval_callback,wandbcallback],#wandbcallback
                    n_eval_episodes=5,log_interval=config["learn_timesteps"])
    run.finish()
    wandb.finish()

if __name__ == '__main__':
    main()