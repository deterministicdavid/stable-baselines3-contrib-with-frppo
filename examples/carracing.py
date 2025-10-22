import multiprocessing
import os
import argparse
import gymnasium as gym

#from stable_baselines3 import PPO
from sb3_contrib import FRPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor


import pynvml
import torch
import random

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
def make_env(env_name: str, seed: int):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _init

def train(env_name: str, log_dir: str, name_prefix: str, n_stack: int):
    selected_device = select_free_gpu_or_fallback()

    # --- Create multiple async environments ---
    num_envs = 6  # you can adjust this (4–12 works well depending on your CPU)
    env_fns = [make_env(env_name=env_name, seed=i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    # --- Add vectorized wrappers ---
    env = VecMonitor(env)               # tracks episode rewards/lengths
    env = VecFrameStack(env, n_stack=n_stack) # stacks 4 frames per env

    # --- Setup what we save --- 
    checkpoint_callback = OverwriteCheckpointCallback(
          save_freq=10000,  # This is now based on total timesteps
          save_path=log_dir,
          name_prefix=name_prefix
        )

    # --- Create and train the model ---
    fr_tau_penalty = 0.01
    model = FRPPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=1024,       # adjust batch sizes to your CPU/GPU
        batch_size=1024,
        fr_penalty_tau=fr_tau_penalty,
        tensorboard_log=log_dir,
        device=selected_device
    )

    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)
    save_file = os.path.join(log_dir, name_prefix)
    model.save(save_file)

    env.close()

def vizualize(env_name: str, model_path: str, n_stack: int):
    print("Starting visualization...")
    
    video_folder = "videos/"
    os.makedirs(video_folder, exist_ok=True)

    # Create a single environment for visualization
    # The render_mode must be "rgb_array" for the RecordVideo wrapper
    env = gym.make(env_name, render_mode="rgb_array")
    # Wrap the environment to record a video
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e == 0)
    
    # Wrap for SB3 and FrameStack (must match training setup)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first.")
        return
        
    model = FRPPO.load(model_path, env=vec_env)
    print(f"Model loaded from {model_path}.")

    # Run one episode
    obs = vec_env.reset()
    done = False
    rewsum = 0
    step = 0
    while not done:
        step += 1
        # Use deterministic actions for evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = vec_env.step(action)
        done = done[0] # we only have one envirnoment
        rewsum += rew[0]
        if (step+1) % 100 == 0:
            print(f"Step: {step+1}, reward so far: {rewsum:.2f}")

    # The video is saved automatically when the environment is closed
    vec_env.close()
    print(f"Visualization complete. Video saved in '{video_folder}' folder.")

if __name__ == '__main__':
    # to fix debugging on MacOS
    # multiprocessing.set_start_method("spawn")

    # --- Add command line argument parsing ---
    parser = argparse.ArgumentParser(description="Train or visualize a FRPPO agent for CarRacing-v3.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    parser.add_argument("--visualise", action="store_true", help="Run the visualization process.")
    args = parser.parse_args()


    n_stack = 4
    env_name = "CarRacing-v3"
    log_dir="./logs/"
    name_prefix = "frppo_carracing_latest"

    if not args.train and not args.visualise:
        print("No action specified. Please use --train or --visualise (or both).")
        parser.print_help()
    else:
        if args.train:
            print("--- Running Training ---")
            train(env_name=env_name, log_dir=log_dir, name_prefix=name_prefix, n_stack=n_stack)
        if args.visualise:
            print("--- Running Visualization ---")
            model_path = os.path.join(log_dir, f"{name_prefix}.zip")
            vizualize(env_name=env_name, model_path=model_path, n_stack=n_stack)

    