import multiprocessing
import os
import argparse
import gymnasium as gym
import ale_py

from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3 import PPO
from sb3_contrib import FRPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor


import pynvml
import torch
import random
import yaml   
import glob
from moviepy import VideoFileClip

class OverwriteCheckpointCallback(BaseCallback):
    """
    Callback for saving a model periodically, overwriting the same file.

    :param save_freq: The frequency (in total timesteps) at which to save the model.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: The name of the file to save the model to.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "latest_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.num_saves = 0
        self.save_path = save_path
        # The full path for the save file, e.g., ./logs/latest_model.zip
        self.save_file = os.path.join(save_path, f"{name_prefix}.zip")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # self.num_timesteps is the total number of steps taken in the environment
        if self.num_timesteps > (self.num_saves+1) * self.save_freq:
            self.num_saves += 1
            self.model.save(self.save_file)
            if self.verbose > 0:
                print(f"Saving latest model to {self.save_file}")
        return True


def select_free_gpu_or_fallback():
    """
    Selects MPS on Arm Macs, on CUDA systems the GPU with the most free memory.
    Returns the device as a torch.device object.
    """
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # Initialize NVML
        pynvml.nvmlInit()

        device_count = torch.cuda.device_count()
        gpu_indices = list(range(device_count))
        random.shuffle(gpu_indices)  # Randomize the order of GPU checks

        best_gpu = None
        max_free_mem = 0

        for i in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = int(meminfo.free)
            if free_mem > max_free_mem:
                best_gpu = i
                max_free_mem = free_mem

        if best_gpu is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{best_gpu}")
        
    print(f"Selected device: {device}")
    return device


# --- Helper function to create environments ---
def make_env_default(env_name: str, seed: int):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _init

def make_env_atari(env_name: str, seed: int):
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
        
        # Apply standard Atari preprocessing
        # This will gray-scale, resize, and handle frame-skipping
        # terminal_on_life_loss=False is default, which is correct for training
        env = AtariPreprocessing(env, grayscale_newaxis=True)
        
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
    
    if env_name.startswith("ALE/"):
        env_fns = [make_env_atari(env_name=env_name, seed=i) for i in range(num_envs)]
    else:
        env_fns = [make_env_default(env_name, seed=i, n_stack=n_stack) for i in range(num_envs)]
        
    env = SubprocVecEnv(env_fns)

    # --- Add vectorized wrappers ---
    env = VecMonitor(env)               # tracks episode rewards/lengths
    env = VecFrameStack(env, n_stack=n_stack) # stacks 4 frames per env

    # --- Setup what we save --- 
    checkpoint_callback = OverwriteCheckpointCallback(
          save_freq=save_freq,  # This is now based on total timesteps
          save_path=log_dir,
          name_prefix=name_prefix
        )

    # --- Create and train the model ---
    
    model = None
    if learning_algo == "FRPPO":
        fr_tau_penalty = config['train']['fr_tau_penalty']
        model = FRPPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=n_steps,       # adjust batch sizes to your CPU/GPU
            batch_size=batch_size,
            fr_penalty_tau=fr_tau_penalty,
            ent_coef=ent_coef,
            tensorboard_log=log_dir,
            device=selected_device
        )
    elif learning_algo == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=n_steps,       # adjust batch sizes to your CPU/GPU
            batch_size=batch_size,
            ent_coef=ent_coef,
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


def post_process_video(input_path: str, output_path: str, scale_factor: int = 4):
    """
    Loads a video, scales it up using nearest-neighbor (pixelated),
    and saves the result.
    """
    try:
        clip = VideoFileClip(input_path)
        
        # Scale the clip
        # interp="nearest" is crucial for the pixel-art look
        scaled_clip = clip.resized(width=clip.w * scale_factor)
        
        # Write the new video file
        scaled_clip.write_videofile(output_path, logger=None)
        
        clip.close()
        scaled_clip.close()
        print(f"Scaled video saved to {output_path}")
        
    except Exception as e:
        print(f"\nError during video post-processing: {e}")
        print("Please ensure 'moviepy' is installed (`pip install moviepy`)")
        print("And that 'ffmpeg' is available on your system.")

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
    MAX_STEPS = 1_000
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

    