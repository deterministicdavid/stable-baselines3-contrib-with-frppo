import os
import pynvml

import torch

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

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
        # random.shuffle(gpu_indices)  # Randomize the order of GPU checks

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
