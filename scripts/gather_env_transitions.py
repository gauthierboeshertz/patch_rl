import argparse

from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
import numpy as np
from gym import wrappers

def gather_env_transitions(env,num_transitions,num_action_repeat=1,data_name="data"):
    states, actions, rewards, next_states, dones = [],[],[],[],[]
    
    state = env.reset()
    
    action = env.action_space.sample()
    action_rep = 0
    for _ in tqdm(range(num_transitions)):
        
        if action_rep == num_action_repeat:
            action = env.action_space.sample() # Should keep the action for some timestep?
            action_rep = 0
        else:
            action_rep += 1
            
        next_state, reward, done, _ = env.step(action)
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
    episode_timesteps = 100
    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],is_demo=False,timeout_steps=episode_timesteps, 
                                                            sparse_reward=True,
                                                            one_sprite_mover=config["one_sprite_mover"],
                                                            all_sprite_mover=config["all_sprite_mover"],
                                                            discrete_all_sprite_mover=config["discrete_all_sprite_mover"],
                                                            random_init_places=config["random_init_places"],
                                                            visual_obs = config["visual_obs"],
                                                            instant_move = config["instant_move"],
                                                            action_scale=0.05)
    

    env = environment.Environment(**env_config)
    gym_env = gym_wrapper.GymWrapper(env)
    
    if config["visual_obs"]:
        num_stack = 3 if not (config["instant_move"] or config["discrete_all_sprite_mover"]) else 1
        gym_env = wrappers.FrameStack(gym_env,num_stack)
    data_name = "../data/{}_{}transitions_{}_{}_{}_{}{}".format("visual" if config["visual_obs"] else "states",config["num_transitions"],config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "discrete_all_sprite_mover" if config["discrete_all_sprite_mover"] else "select_move"),config["random_init_places"],config["num_action_repeat"],"instantmove" if config["instant_move"] else "")
    print("Data name",data_name)
    gather_env_transitions(gym_env,config["num_transitions"],num_action_repeat=config["num_action_repeat"],data_name=data_name)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sprites', type=int, default=1)
    parser.add_argument('--one_sprite_mover', action="store_true")
    parser.add_argument('--all_sprite_mover', action="store_true")
    parser.add_argument('--discrete_all_sprite_mover', action="store_true")
    parser.add_argument('--num_transitions', type=int,default=20000)
    parser.add_argument('--random_init_places', action="store_true")
    parser.add_argument('--visual_obs', action="store_true")
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('--num_action_repeat', type=int,default=1)
    parser.add_argument('--instant_move', action="store_true")

    main(vars(parser.parse_args()))
    