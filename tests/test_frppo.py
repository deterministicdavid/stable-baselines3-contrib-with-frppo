# tests/test_my_ppo_variant.py

import pytest
import gymnasium as gym
import numpy as np

from sb3_contrib import FRPPO



def test_my_ppo_variant():
    """Test that MyPPOVariant can be instantiated and run"""
    env = gym.make("CartPole-v1")
    model = FRPPO("MlpPolicy", env, n_steps=128, verbose=0)
    model.learn(total_timesteps=256)


def test_my_ppo_variant_prediction():
    """Test that the model can make predictions"""
    env = gym.make("CartPole-v1")
    model = FRPPO("MlpPolicy", env, n_steps=128, verbose=0)
    model.learn(total_timesteps=256)
    
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action is not None


def test_my_ppo_variant_save_load():
    """Test saving and loading"""
    env = gym.make("CartPole-v1")
    model = FRPPO("MlpPolicy", env, n_steps=128, verbose=0)
    model.learn(total_timesteps=256)
    
    model.save("test_my_ppo_variant")
    loaded_model = FRPPO.load("test_my_ppo_variant", env=env)
    
    obs, _ = env.reset()
    action1, _ = model.predict(obs, deterministic=True)
    action2, _ = loaded_model.predict(obs, deterministic=True)
    assert action1 == action2


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_my_ppo_variant_different_envs(env_id):
    """Test on different environment types"""
    env = gym.make(env_id)
    model = FRPPO("MlpPolicy", env, n_steps=128, verbose=0)
    model.learn(total_timesteps=256)


def test_custom_parameter():
    """Test that custom parameter works"""
    env = gym.make("CartPole-v1")
    custom_value = 1e-2
    model = FRPPO("MlpPolicy", env, fr_penalty_tau=custom_value, verbose=0)
    assert model.fr_penalty_tau.value_schedule.val == custom_value