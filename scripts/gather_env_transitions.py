import argparse

from moog import environment
from moog.env_wrappers import gym_wrapper
from moog_demos.example_configs import bouncing_sprites
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm
import numpy as np
from gym import wrappers
import hydra
from stable_baselines3 import TD3

def gather_env_transitions(env,num_transitions,trained_agent=None,num_action_repeat=1,data_name="data"):
    states, actions, rewards, next_states, dones = [],[],[],[],[]
    
    state = env.reset()
    
    action = env.action_space.sample()
    action_rep = 0
    ep_reward = 0
    for _ in tqdm(range(num_transitions)):
        
        if trained_agent is  None:
            if action_rep == num_action_repeat:
                action = env.action_space.sample() # Should keep the action for some timestep?
                action_rep = 0
            else:
                action_rep += 1
        else:
            action = trained_agent.predict(state)[0][0]
            action = np.clip(action + np.random.normal(0,0.5,size=action.shape).astype(np.float32), env.action_space.low+0.00001, env.action_space.high-0.00001)
            
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        ep_reward += reward
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        state = next_state
        if done:
            
            print("Episode reward",ep_reward)
            ep_reward = 0
            state = env.reset()
    
    print("Cutoff reward",ep_reward)
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)
        
          
    print("States shape",states.shape)
    
    np.savez_compressed(data_name,states=states,actions=actions,rewards=rewards,next_states=next_states,dones=dones)

@hydra.main(config_path="configs", config_name="gather_transitions")
def main(config):
    

    num_action_repeat = config["num_action_repeat"]
    trained_agent_path = config["trained_agent_path"]
    num_transitions = config["num_transitions"]
    
    config = config["env"]
    
    set_random_seed(config["seed"])
    episode_timesteps = 100
    env_config = bouncing_sprites.get_config(num_sprites=config["num_sprites"],
                                             is_demo=False,
                                             timeout_steps=episode_timesteps, 
                                            contact_reward=True,
                                            one_sprite_mover=config["one_sprite_mover"],
                                            all_sprite_mover=config["all_sprite_mover"],
                                            discrete_all_sprite_mover=config["discrete_all_sprite_mover"],
                                            random_init_places=config["random_init_places"],
                                            visual_obs = config["visual_obs"],
                                            instant_move = config["instant_move"],
                                            dont_show_targets=config["dont_show_targets"],
                                            disappear_after_contact=True,
                                            seed = config["seed"],
                                            action_scale=0.05)
    

    env = environment.Environment(**env_config)
    gym_env = gym_wrapper.GymWrapper(env)
    
    if config["visual_obs"]:
        num_stack = 3 if not (config["instant_move"] or config["discrete_all_sprite_mover"]) else 1
        gym_env = wrappers.FrameStack(gym_env,num_stack)
    data_name = "../data/{}_{}_{}transitions_{}_{}_{}_{}{}{}".format("expert" if trained_agent_path else "","visual" if config["visual_obs"] else "states",num_transitions,config["num_sprites"],("all_sprite_mover"if config["all_sprite_mover"] else "one_sprite_mover" if config["one_sprite_mover"] else "discrete_all_sprite_mover" if config["discrete_all_sprite_mover"] else "select_move"),config["random_init_places"],num_action_repeat,"instantmove" if config["instant_move"] else "", "no_targets"if config["dont_show_targets"] else "")
    print("Data name",data_name)
    
    if trained_agent_path:
        agent = TD3.load(trained_agent_path)
    else:
        agent = None #random agent
        
    gather_env_transitions(gym_env,num_transitions,
                            num_action_repeat=num_action_repeat,
                            data_name=data_name, trained_agent=agent)

    
if __name__ == "__main__":

    main()
    