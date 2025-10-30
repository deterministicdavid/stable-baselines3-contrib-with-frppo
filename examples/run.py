import os
import argparse
import gymnasium as gym
import ale_py
import numpy as np

from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import PPO
from sb3_contrib import FRPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor


import glob
import torch
import random
import yaml   

from run_utils import OverwriteCheckpointCallback, select_free_gpu_or_fallback, post_process_video
from own_policy import CustomActorCriticCnnPolicy

class ClipRewardWrapper(gym.Wrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info

# --- Helper function to create environments ---
def make_env_default(env_name: str, seed: int):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _init

def make_env_atari(env_name: str, seed: int, bin_rewards=True):
    """
    Utility function for multiprocessed env creation.
    
    :param env_name: the environment ID
    :param seed: the initial seed for RNG
    :param n_stack: Number of frames to stack
    """
    def _init():
        # Add frameskip=1 to disable internal frame skipping
        env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
        env.reset(seed=seed)
        
        

        """
        We try to follow "The 37 implementation details of Proximal Policy Optimization"
        - Using noop_max=30, which is the default, is #1 
        - Setting frame_skip=4 and using max pooling (which it does if frame_skip>1) is #2
        - Setting terminal_on_life_loss=True is #3. 
        - By default the AtariPreprocessing will resize to 84x84, this is #5.
        """
        # Setting scale_obs=True is 9 of "The 37 implementation details of Proximal Policy Optimization"
        # But when using CnnPolicy SB3 does this anyway and our custom policy does that too, so scale_obs=False
        # should be fine.
        # Using frame_skip=4 
        env = AtariPreprocessing(env, noop_max=30, grayscale_newaxis=True, terminal_on_life_loss=True, frame_skip=4, scale_obs=False)
        
        if bin_rewards:
            # This is number #6 of "The 37 implementation details of Proximal Policy Optimization"
            # Will bin rewards to the bins {-1,0,+1}
            env = ClipRewardWrapper(env)

        # Record stats
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
    return _init

def train(config: dict):
    selected_device = select_free_gpu_or_fallback()

    learning_algo = config['train']['algo']
    env_name = config['env_name']
    n_stack = config['n_stack']
    num_envs = config['train']['n_envs']
    log_dir = config['logging']['log_dir']
    name_prefix = config['logging']['name_prefix']
    save_freq = config['logging']['checkpoint_save_freq']
    n_steps = config['train']['n_steps']
    batch_size = config['train']['batch_size']
    total_timesteps = config['train']['total_timesteps']
    ent_coef = config['train']['ent_coef']
    
    # Note that using CnnPolicy makes SB3 normalize images to [0,1], 
    # which is #9 of "The 37 implementation details of Proximal Policy Optimization"
    policy = "CnnPolicy"
    if config['train']['policy'] == "own":
        policy = CustomActorCriticCnnPolicy # this will also normalize to [0,1]

    if env_name.startswith("ALE/"):
        env_fns = [make_env_atari(env_name=env_name, seed=i) for i in range(num_envs)]
    else:
        env_fns = [make_env_default(env_name, seed=i) for i in range(num_envs)]
        
    env = SubprocVecEnv(env_fns)

    # --- Add vectorized wrappers ---
    env = VecMonitor(env)               
    
    # Using frame stacking with n_stack=4 is #7 of "The 37 implementation details of Proximal Policy Optimization"
    env = VecFrameStack(env, n_stack=n_stack) 

    # --- Setup what we save --- 
    checkpoint_callback = OverwriteCheckpointCallback(
          save_freq=save_freq,  # This is now based on total timesteps
          save_path=log_dir,
          name_prefix=name_prefix
        )

    # --- Create and train the model ---
    
    
    
    model = None
    learning_rate = 3e-4 
    if config['train']['decay_lr']:
        learning_rate = lambda f : f * 2.5e-4
    
    default_n_opt_epochs = 4
    n_opt_epochs = config.get('train', {}).get('n_opt_epochs', default_n_opt_epochs)

    if learning_algo == "FRPPO":
        fr_tau_penalty = config['train']['fr_tau_penalty']
        fr_penalty_scale_by_adv = config['train']['fr_scale_by_adv']
        model = FRPPO(
            policy=policy,
            env=env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,       
            n_epochs=n_opt_epochs,
            batch_size=batch_size,
            fr_penalty_tau=fr_tau_penalty,
            fr_penalty_scale_by_adv=fr_penalty_scale_by_adv,
            ent_coef=ent_coef,
            tensorboard_log=log_dir,
            device=selected_device
        )
    elif learning_algo == "PPO":
        default_clip_epsilon = 0.2 
        clip_epsilon = config.get('train', {}).get('clip_epsilon', default_clip_epsilon)
        model = PPO(
            policy=policy,
            env=env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps, 
            n_epochs=n_opt_epochs,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_epsilon,
            tensorboard_log=log_dir,
            device=selected_device
        )
    else:
        print(f"Learning algorithm {learning_algo} may be in SB3 but not it's not been setup here.")
        return

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    save_file = os.path.join(log_dir, name_prefix)
    model.save(save_file)

    env.close()



def vizualize(config: dict):
    print("Starting visualization...")
    
    learning_algo = config['train']['algo']
    env_name = config['env_name']
    n_stack = config['n_stack']
    video_folder = config['visualize']['video_folder']
    log_dir = config['logging']['log_dir']
    name_prefix = config['logging']['name_prefix']
    deterministic_actions = config['visualize']['deterministic']
    model_path = os.path.join(log_dir, f"{name_prefix}.zip")
    

    os.makedirs(video_folder, exist_ok=True)



    # --- Helper function to create the visualization env ---
    def make_viz_env():
        """Creates and wraps the environment for visualization."""
        if env_name.startswith("ALE/"):
            print("Atari environment detected. Using Atari-specific wrappers for visualization.")
            # Add frameskip=1 to disable internal frame skipping for Atari
            env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
            # 1. Wrap for video *first* to capture the full-resolution render
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e == 0)
            # Apply Atari preprocessing
            # *** terminal_on_life_loss=True ***
            # This is the key fix: it makes the env send `done=True` after one life is lost.
            # grayscale_newaxis=True adds channel dim for CnnPolicy compatibility
            env = AtariPreprocessing(env, terminal_on_life_loss=True, grayscale_newaxis=True)
        else:
            print("Default environment detected.")
            # Non-Atari env
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e == 0)

        
        return env
    
    # Wrap for SB3 and FrameStack (must match training setup)
    # Pass the *function* to DummyVecEnv, not an already-created object
    vec_env = DummyVecEnv([make_viz_env])
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first.")
        return
        
    model = None
    if learning_algo == "FRPPO":
        model = FRPPO.load(model_path, env=vec_env)
    elif learning_algo == "PPO":
        model = PPO.load(model_path, env=vec_env)
    else: 
        print(f"Learning algorithm {learning_algo} may be in SB3 but not it's not been setup here.")
        return

    print(f"Model loaded from {model_path}.")

    model.set_env(vec_env)

    # Run one episode
    obs = vec_env.reset()
    done = False
    rewsum = 0
    step = 0
    MAX_STEPS = 100_000
    while not done:
        step += 1
        # Use deterministic actions for evaluation
        action, _ = model.predict(obs, deterministic=deterministic_actions)
        obs, rew, done, info = vec_env.step(action)
        done = done[0] # we only have one envirnoment
        rewsum += rew[0]
        if (step+1) % 100 == 0:
            print(f"Step: {step+1}, reward so far: {rewsum:.2f}")
        if step > MAX_STEPS:
            break

    # The video is saved automatically when the environment is closed
    vec_env.close()
    print(f"Visualization complete. Video saved in '{video_folder}' folder. Total steps is {step}. Total reward is {rewsum}.")

    # --- ADD THIS LOGIC ---
    # Find the most recently created video file in the folder
    list_of_files = glob.glob(os.path.join(video_folder, '*.mp4'))
    if not list_of_files:
        print("Error: No video file found to post-process.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    output_file = os.path.join(video_folder, f"scaled_{os.path.basename(latest_file)}")

    # Scale the video
    post_process_video(latest_file, output_file, scale_factor=4)


if __name__ == '__main__':
    # to fix debugging on MacOS
    # multiprocessing.set_start_method("spawn")

    # --- Add command line argument parsing ---
    parser = argparse.ArgumentParser(description="Train or visualize a FRPPO agent for CarRacing-v3.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    parser.add_argument("--visualise", action="store_true", help="Run the visualization process.")
    parser.add_argument("--numenvs", type=int, default=6, help="Number of parallel environments to use for training.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        parser.print_help()
        exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not args.train and not args.visualise:
        print("No action specified. Please use --train or --visualise (or both).")
        parser.print_help()
    else:
        if args.train:
            print(f"--- Running Training --- up to a total of {config['train']['total_timesteps']}")
            train(config=config)
        if args.visualise:
            print("--- Running Visualization ---")
            vizualize(config=config)

    