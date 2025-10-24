import torch
import torch.nn as nn  
import numpy as np     


from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

# --- Helper function for network initialization ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear or convolutional layer with orthogonal initialization.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# --- Custom Policy with Separate Actor/Critic Networks ---
class CustomActorCriticCnnPolicy(ActorCriticPolicy):
    """
    Custom policy for PPO with separate CNNs for actor and critic.
    Based on the architecture from the user's request.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        **kwargs,
    ):
        # --- FIX 1: Set a flag to defer optimizer creation ---
        self._networks_defined = False
        
        # --- Call super().__init__() normally ---
        # This call *will* trigger self.build_optimizer() below.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
        
        # --- Deactivate default network attributes ---
        # We must set them to None so the parent class doesn't try to use them
        self.features_extractor = None
        self.mlp_extractor = None
        self.action_net = None
        self.value_net = None


        # The observation space is already (C, H, W) = (4, 84, 84)
        # due to the VecTransposeImage wrapper
        n_input_channels = observation_space.shape[0]
        
        # --- Define the CNN base ---
        # This function creates one copy of the CNN base
        def create_cnn_base():
            return nn.Sequential(
                # Input shape is (N, C, H, W) = (N, 4, 84, 84)
                layer_init(nn.Conv2d(n_input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                # Input features to linear layer: 64 * 7 * 7 = 3136
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )

        # --- Actor Network (Policy) ---
        self.actor_cnn = create_cnn_base()
        self.actor_head = layer_init(nn.Linear(512, action_space.n), std=0.01)
        
        # --- Critic Network (Value) ---
        self.critic_cnn = create_cnn_base()
        self.critic_head = layer_init(nn.Linear(512, 1), std=1)

        # --- FIX 2: Manually build the optimizer *after* networks are created ---
        self._networks_defined = True
        self.build_optimizer(lr_schedule)

    def build_optimizer(self, lr_schedule) -> None:
        """
        Override the build_optimizer method.
        This method is called by super().__init__(), but we use a flag
        to prevent it from running until our custom networks are defined.
        """
        # --- FIX 3: Defer the optimizer creation ---
        if not getattr(self, "_networks_defined", False):
            return  # Do nothing if called from super().__init__
        
        # When called manually from self.__init__(), proceed as normal.
        # This will correctly gather params from self.actor_cnn etc.
        # --- REPLICATE BasePolicy.build_optimizer ---
        # This avoids the super() call that was causing the AttributeError
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Helper function to get the action distribution from the actor's latent features.
        """
        logits = self.actor_head(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        """
        Forward pass for prediction (action selection).
        """
        # --- FIX 2: Cast to float, normalize, and remove permute ---
        # observation is already (N, C, H, W) and uint8
        obs_float = observation.float() / 255.0
        
        latent_pi = self.actor_cnn(obs_float) # <-- Pass float tensor
        distribution = self._get_action_dist_from_latent(latent_pi)
        action = distribution.get_actions(deterministic=deterministic)
        return action

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass for `model.predict()`
        """
        # --- FIX 3 (part a): Cast to float, normalize, and remove permute ---
        # obs is already (N, C, H, W) and uint8
        obs_float = obs.float() / 255.0
        
        # Actor
        latent_pi = self.actor_cnn(obs_float) # <-- Pass float tensor
        distribution = self._get_action_dist_from_latent(latent_pi)
        action = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(action)
        
        # Critic
        latent_vf = self.critic_cnn(obs_float) # <-- Pass float tensor
        value = self.critic_head(latent_vf)
        
        return action, value, log_prob
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for predicting values (used in collect_rollouts).
        """
        # obs is already (N, C, H, W) and uint8
        obs_float = obs.float() / 255.0
        
        # Critic
        latent_vf = self.critic_cnn(obs_float) # <-- Pass float tensor
        value = self.critic_head(latent_vf)
        return value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Forward pass for training (used in `model.train()`).
        """
        # --- FIX 3 (part b): Cast to float, normalize, and remove permute ---
        # obs is already (N, C, H, W) and uint8
        obs_float = obs.float() / 255.0
        
        # Actor
        latent_pi = self.actor_cnn(obs_float) # <-- Pass float tensor
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Critic
        latent_vf = self.critic_cnn(obs_float) # <-- Pass float tensor
        value = self.critic_head(latent_vf)
        
        return value, log_prob, entropy