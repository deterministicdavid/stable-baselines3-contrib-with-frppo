import os
import multiprocessing as mp
import argparse
import copy

import gymnasium as gym
import ale_py
import numpy as np

from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from stable_baselines3 import PPO
from sb3_contrib import FRPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecVideoRecorder,
    VecNormalize,
)
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.env_util import make_atari_env, make_vec_env


import glob
import torch
import random
import yaml

from run_utils import (
    OverwriteCheckpointCallback,
    select_free_gpu_or_fallback,
    get_free_cuda_gpus,
    post_process_video,
    log_hyper_parameters,
)
from own_policy import CustomActorCriticCnnPolicy


# --- Helper function to create environments ---
def make_env_default(env_name: str, seed: int):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return _init


def make_env_mujoco(config: dict, seed: int):
    def _init():
        env = gym.make(config["env_name"])
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10.0, 10.0), None)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10.0, 10.0))

        # env.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init


def train(config: dict, assigned_device: torch.device, seed: int):

    learning_algo = config["train"]["algo"]

    env_name = config["env_name"]
    env_is_atari = config.get("env_is_atari", True)
    env_is_mujoco = config.get("env_is_mujoco", False)
    assert not (env_is_atari and (not env_is_mujoco))

    n_stack = config["n_stack"]
    num_envs = config["train"]["n_envs"]
    log_dir = config["logging"]["log_dir"]
    name_prefix = config["logging"]["name_prefix"]
    save_freq = config["logging"]["checkpoint_save_freq"]
    n_steps = config["train"]["n_steps"]
    batch_size = config["train"]["batch_size"]
    total_timesteps = config["train"]["total_timesteps"]
    ent_coef = config["train"]["ent_coef"]

    policy = None
    policy_kwargs = {}
    never_mps = False
    if env_is_atari:
        # Note that using CnnPolicy makes SB3 normalize images to [0,1],
        # which is #9 of "The 37 implementation details of Proximal Policy Optimization"
        policy = "CnnPolicy"
        policy_kwargs = {
            "ortho_init": True,
            "features_extractor_class": NatureCNN,
            "share_features_extractor": True,
            "normalize_images": True,
        }

        if config["train"]["policy"] == "own":
            policy = CustomActorCriticCnnPolicy  # this will also normalize to [0,1]

        env = make_atari_env(
            env_id=env_name,
            n_envs=num_envs,
            seed=seed,
            wrapper_kwargs={
                "noop_max": 30,
                "frame_skip": 4,
                "screen_size": 84,
                "terminal_on_life_loss": True,
                "clip_reward": True,
                "action_repeat_probability": 0.0,
            },
            vec_env_cls=DummyVecEnv,
        )
        # Using frame stacking with n_stack=4 is #7 of "The 37 implementation details of Proximal Policy Optimization"
        env = VecFrameStack(env, n_stack=n_stack)

    elif env_is_mujoco:
        policy = "MlpPolicy"  # TODO: check whether we need to set more options to track "The 37 implementation details"
        policy_kwargs = {
            "ortho_init": True,
            "share_features_extractor": False,
        }

        def custom_wrappers(env: gym.Env) -> gym.Env:
            # env = gym.wrappers.ClipAction(env) # this messes up the action space
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
            return env

        env = make_vec_env(env_name, n_envs=num_envs, wrapper_class=custom_wrappers)
        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        if n_stack > 0:
            print(f'Mujoco doesn\'t do "frame" stacking.')
    else:
        policy = "CnnPolicy"  # TODO: this will need more work as it won't work for both car racing and the classic stuff
        env_fns = [make_env_default(env_name, seed=i) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env)
        env = VecFrameStack(env, n_stack=n_stack)

    # --- Setup what we save ---
    checkpoint_callback = OverwriteCheckpointCallback(
        save_freq=save_freq, save_path=log_dir, name_prefix=name_prefix  # This is now based on total timesteps
    )

    # --- Create and train the model ---

    model = None
    learning_rate = 3e-4
    if config["train"]["decay_lr"]:
        lr_decay_init = float(config.get("train", {}).get("decay_lr_init", "2.5e-4"))
        learning_rate = lambda f: f * lr_decay_init

    default_n_opt_epochs = 4
    n_opt_epochs = config.get("train", {}).get("n_opt_epochs", default_n_opt_epochs)

    print("Action space:", env.action_space)
    print("Low:", env.action_space.low)
    print("High:", env.action_space.high)

    if learning_algo == "FRPPO":
        fr_tau_penalty = config["train"]["fr_tau_penalty"]
        fr_penalty_scale_by_adv = config["train"]["fr_scale_by_adv"]
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
            device=assigned_device,
            policy_kwargs=policy_kwargs,
        )
    elif learning_algo == "PPO":
        default_clip_epsilon = 0.2
        clip_epsilon = config.get("train", {}).get("clip_epsilon", default_clip_epsilon)
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
            device=assigned_device,
            policy_kwargs=policy_kwargs,
        )
    else:
        print(f"Learning algorithm {learning_algo} may be in SB3 but not it's not been setup here.")
        return

    tb_path = os.path.join(log_dir, f"{model.__class__.__name__}_{config['train']['run_id']}")
    os.makedirs(tb_path, exist_ok=True)
    new_logger = configure(tb_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    log_hyper_parameters(logger=model.logger, config=config)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    save_file = os.path.join(log_dir, name_prefix)
    model.save(save_file)

    env.close()


def vizualize(config: dict, name_override=None):
    print("Starting visualization...")
    MAX_STEPS = 10_000

    learning_algo = config["train"]["algo"]
    env_name = config["env_name"]
    env_is_atari = config.get("env_is_atari", True)
    n_stack = config["n_stack"]
    video_folder = config["visualize"]["video_folder"]
    log_dir = config["logging"]["log_dir"]
    if name_override == "":
        name_prefix = config["logging"]["name_prefix"]
    else:
        name_prefix = name_override
    deterministic_actions = config["visualize"]["deterministic"]
    seed = None  # random visualisations
    model_path = os.path.join(log_dir, f"{name_prefix}.zip")

    os.makedirs(video_folder, exist_ok=True)

    if env_is_atari:
        print("Atari environment detected. Using Atari-specific wrappers for visualization.")
        env = make_atari_env(
            env_id=env_name,
            n_envs=1,
            seed=seed,
            wrapper_kwargs={
                "noop_max": 30,
                "frame_skip": 4,
                "screen_size": 84,
                "terminal_on_life_loss": True,
                "clip_reward": True,
                "action_repeat_probability": 0.0,
            },
            vec_env_cls=DummyVecEnv,
        )
        env = VecFrameStack(env, n_stack=n_stack)
    else:
        print("Default environment detected.")
        # Non-Atari env
        # env = gym.make(env_name, render_mode="rgb_array")
        env = make_vec_env(env_name, n_envs=1)

    env = VecVideoRecorder(
        venv=env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 0,  # record first episode
        video_length=MAX_STEPS,
        name_prefix=f"{learning_algo}",
    )

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first.")
        return

    model = None
    if learning_algo == "FRPPO":
        model = FRPPO.load(model_path, env=env)
    elif learning_algo == "PPO":
        model = PPO.load(model_path, env=env)
    else:
        print(f"Learning algorithm {learning_algo} may be in SB3 but not it's not been setup here.")
        return

    print(f"Model loaded from {model_path}.")

    model.set_env(env)

    # Run one episode
    obs = env.reset()
    done = False
    rewsum = 0
    step = 0

    while not done:
        step += 1
        # Use deterministic actions for evaluation
        action, _ = model.predict(obs, deterministic=deterministic_actions)
        obs, rew, done, info = env.step(action)
        done = done[0]  # we only have one envirnoment
        rewsum += rew[0]
        if (step + 1) % 100 == 0:
            print(f"Step: {step+1}, reward so far: {rewsum:.2f}")
        if step > MAX_STEPS:
            break

    # The video is saved automatically when the environment is closed
    env.close()
    print(f"Visualization complete. Video saved in '{video_folder}' folder. Total steps is {step}. Total reward is {rewsum}.")

    # TODO: hide this behind a flag
    if False:
        # Find the most recently created video file in the folder
        list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        if not list_of_files:
            print("Error: No video file found to post-process.")
            return

        latest_file = max(list_of_files, key=os.path.getctime)
        output_file = os.path.join(video_folder, f"scaled_{os.path.basename(latest_file)}")

        # Scale the video
        post_process_video(latest_file, output_file, scale_factor=4)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # --- Add command line argument parsing ---
    parser = argparse.ArgumentParser(description="Train or visualize a FRPPO agent for CarRacing-v3.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    parser.add_argument("--visualise", action="store_true", help="Run the visualization process.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument(
        "--parallel_runs",
        type=int,
        default=1,
        help="Maximum number of parallel training runs. Will be limited by the number of free GPUs.",
    )
    parser.add_argument(
        "--vis_model_name",
        type=str,
        default="",
        help="Model prefix name (without the .zip, without the name of the log directory from config)",
    )

    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        parser.print_help()
        exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not args.train and not args.visualise:
        print("No action specified. Please use --train or --visualise (or both).")
        parser.print_help()
    else:
        if args.train:
            print(f"--- Preparing for up to {args.parallel_runs} parallel training run(s) ---")
            remaining_parallel_runs = args.parallel_runs
            seed = config["train"].get("random_seed", None)
            force_cpu = config["train"].get("force_cpu", False)
            while remaining_parallel_runs > 0:
                gpus_to_use = get_free_cuda_gpus(max_count=args.parallel_runs, force_cpu=force_cpu)
                num_to_run = min(remaining_parallel_runs, len(gpus_to_use))
                print(
                    f"Found {len(gpus_to_use)} free GPUs. Launching {num_to_run} out of remaining {remaining_parallel_runs} run(s) on up to {len(gpus_to_use)} GPUs."
                )
                processes = []
                for device_idx in range(0, num_to_run):
                    device = gpus_to_use[device_idx]
                    # copy and overwrite config
                    run_config = copy.deepcopy(config)
                    original_run_id = run_config["train"]["run_id"]
                    original_prefix = run_config["logging"]["name_prefix"]
                    run_config["train"]["run_id"] = f"{original_run_id}_{device_idx}_gpu_{device.type}_{device.index}"
                    run_config["logging"][
                        "name_prefix"
                    ] = f"{original_prefix}_{device_idx}_gpu_{device.index}"  # this is where the train model gets saved

                    # 4. Create and start the process
                    p = mp.Process(target=train, args=(run_config, device, seed))
                    processes.append(p)
                    p.start()
                    if seed is not None:
                        seed += 1

                # 5. Wait for all processes to finish
                for p in processes:
                    p.join()

                remaining_parallel_runs -= len(gpus_to_use)

        if args.visualise:
            print("--- Running Visualization ---")
            vizualize(config=config, name_override=args.vis_model_name)
