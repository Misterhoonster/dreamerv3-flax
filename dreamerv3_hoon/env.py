from typing import Dict, Tuple

from chex import Array, ArrayTree
import jax
import jax.numpy as jnp
from gymnax.environments.bsuite import catch
import gymnax

class CatchEnv:
    """Catch environment from Gymnax."""

    def __init__(self, seed: int = 0):
        """Initializes an environment."""
        # Environment
        self.env, self.env_params = gymnax.make("Catch-bsuite")
        self.key = jax.random.PRNGKey(seed)
        
        # Get the observation and action spaces
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_space = self.env.action_space(self.env_params)
        
        # Reset variables
        self._done = True
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self) -> Tuple[Array, Dict]:
        """Resets the environment."""
        # Split the key
        self.key, reset_key = jax.random.split(self.key)
        
        # Reset the environment
        obs, state = self.env.reset(reset_key, self.env_params)
        
        # Reset variables
        self._done = False
        self._episode_return = 0.0
        self._episode_length = 0
        
        # Create info dict
        info = {
            "episode_return": self._episode_return,
            "episode_length": self._episode_length,
        }
        
        return obs, info

    def step(self, action: int) -> Tuple[Array, float, bool, Dict]:
        """Steps the environment."""
        if self._done:
            # Reset if done
            obs, info = self.reset()
            return obs, 0.0, False, info
            
        # Split the key
        self.key, step_key = jax.random.split(self.key)
        
        # Step the environment
        obs, state, reward, done, info = self.env.step(
            step_key, state, action, self.env_params
        )
        
        # Update episode stats
        self._episode_return += reward
        self._episode_length += 1
        self._done = done
        
        # Update info
        info = {
            "episode_return": self._episode_return,
            "episode_length": self._episode_length,
        }
        
        return obs, reward, done, info


class VecCatchEnv:
    """Vectorized Catch environment."""
    
    def __init__(self, num_envs: int = 1, seed: int = 0):
        """Initializes vectorized environments."""
        # Environment
        self.env, self.env_params = gymnax.make("Catch-bsuite")
        self.num_envs = num_envs
        
        # Create a separate key for each environment
        self.keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        
        # Get spaces
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_space = self.env.action_space(self.env_params)
        
        # Vectorize reset and step
        self.v_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.v_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))
        
        # Initialize states
        self.obs, self.states = self.v_reset(self.keys, self.env_params)
        self.dones = jnp.zeros(num_envs, dtype=bool)
        
        # Episode stats
        self.episode_returns = jnp.zeros(num_envs)
        self.episode_lengths = jnp.zeros(num_envs)

    def reset(self) -> Tuple[Array, Dict]:
        """Resets all environments."""
        # Reset all environments
        self.keys = jax.random.split(self.keys[0], self.num_envs)
        self.obs, self.states = self.v_reset(self.keys, self.env_params)
        
        # Reset episode stats
        self.dones = jnp.zeros(self.num_envs, dtype=bool)
        self.episode_returns = jnp.zeros(self.num_envs)
        self.episode_lengths = jnp.zeros(self.num_envs)
        
        # Create info dict
        info = {
            "episode_return": self.episode_returns,
            "episode_length": self.episode_lengths,
        }
        
        return self.obs, info

    def step(self, actions: Array) -> Tuple[Array, Array, Array, Dict]:
        """Steps all environments."""
        # Generate new keys
        self.keys = jax.random.split(self.keys[0], self.num_envs)
        
        # Step all environments
        self.obs, self.states, rewards, self.dones, _ = self.v_step(
            self.keys, self.states, actions, self.env_params
        )
        
        # Update episode stats
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        # Reset completed episodes
        reset_obs, reset_states = self.v_reset(self.keys, self.env_params)
        self.obs = jnp.where(self.dones[:, None, None], reset_obs, self.obs)
        self.states = jax.tree_map(
            lambda x, y: jnp.where(self.dones, x, y),
            reset_states,
            self.states
        )

        # print("after tree map:", self.states)
        # print("episode returns:", self.episode_returns)
        # print("episode lengths:", self.episode_lengths)

        # Create info dict
        infos = {
            "episode_return": jnp.where(self.dones, self.episode_returns, 0.0),
            "episode_length": jnp.where(self.dones, self.episode_lengths, 0.0),
        }
        
        # Reset episode stats for completed episodes
        self.episode_returns = jnp.where(self.dones, 0.0, self.episode_returns)
        self.episode_lengths = jnp.where(self.dones, 0.0, self.episode_lengths)

        firsts = self.get_firsts(infos)
        
        return self.obs, rewards, self.dones, firsts, infos

    @staticmethod
    def transform_dones(dones: Array) -> Array:
        """Transforms dones to float32."""
        return dones.astype(jnp.float32)

    @staticmethod
    def get_firsts(infos: ArrayTree) -> Array:
        """Returns firsts based on episode lengths."""
        lengths = infos["episode_length"]
        return (lengths == 0).astype(jnp.float32)
