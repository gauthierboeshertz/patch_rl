from operator import ne
import sys
import argparse
import importlib
import os
from textwrap import wrap

from moog import env_wrappers
from moog import observers
from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
import numpy as np
from gym import wrappers

def gather_env_transitions(env,num_transitions,data_name="data"):
    states, actions, rewards, next_states, dones = [],[],[],[],[]
    
    state = env.reset()
    
    for _ in tqdm(range(num_transitions)):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        state = next_state
        if done:
            state = env.reset()
            
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)
    
    print("States shape",states.shape)
    
    np.savez_compressed(data_name,states=states,actions=actions,rewards=rewards,next_states=next_states,dones=dones)
    
def main(config):
    
    set_random_seed(config["seed"])
    episode_timesteps = 50
    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=False,timeout_steps=episode_timesteps, 
                                                            sparse_reward=True,
                                                            one_sprite_mover=config["one_sprite_mover"],
                                                            all_sprite_mover=config["all_sprite_mover"],
                                                            random_init_places=config["random_init_places"],
                                                            visual_obs = config["visual_obs"])
    

    env = environment.Environment(**env_config)
    gym_env = gym_wrapper.GymWrapper(env)
    
    if config["visual_obs"]:
        gym_env = wrappers.FrameStack(gym_env,3)
    data_name = "transition_data/{}transitions_{}_{}_{}".format(config["num_transitions"],config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "select_move"),config["random_init_places"])
    print("Data name",data_name)
    gather_env_transitions(gym_env,config["num_transitions"],data_name=data_name)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--one_sprite_mover', action="store_true")
    parser.add_argument('--all_sprite_mover', action="store_true")
    parser.add_argument('--num_transitions', type=int,default=20000)
    parser.add_argument('--random_init_places', action="store_true")
    parser.add_argument('--visual_obs', action="store_true")
    parser.add_argument('--seed', type=int,default=0)

    main(vars(parser.parse_args()))
    