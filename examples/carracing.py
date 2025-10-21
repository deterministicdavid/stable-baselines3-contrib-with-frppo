import multiprocessing
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

import pynvml
import torch
import random

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
def make_env(seed: int):
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _init

if __name__ == '__main__':
    # to fix debugging on MacOS
    multiprocessing.set_start_method("spawn")

    selected_device = select_free_gpu_or_fallback()

    # --- Create multiple async environments ---
    num_envs = 1  # you can adjust this (4–12 works well depending on your CPU)
    env_fns = [make_env(seed=i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    # --- Add vectorized wrappers ---
    env = VecMonitor(env)               # tracks episode rewards/lengths
    env = VecFrameStack(env, n_stack=4) # stacks 4 frames per env

    # --- Create and train the model ---
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=1024,       # adjust batch sizes to your CPU/GPU
        batch_size=256,
        tensorboard_log="./logs/",
        device=selected_device
    )

    model.learn(total_timesteps=2_000_000)
    model.save("ppo_carracing_async_stack")

    env.close()
