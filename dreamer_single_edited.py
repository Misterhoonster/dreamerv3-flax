import jax
import jax.numpy as jnp
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.linen import Module
from jax.tree_util import tree_map
from typing import Any, Dict, Tuple, Sequence, Optional

from dreamerv3_flax.head import MLPHead
from dreamerv3_flax.normalizer import Normalizer

# ------------------------------------------------------------------------------
#  1) A purely functional Policy module with no loss-related code
# ------------------------------------------------------------------------------

class Policy(Module):
    """
    Purely functional Policy.
    Only forward methods here—no loss or state mutation.
    """
    num_actions: int
    action_head_kwargs: Dict = FrozenDict(
        hid_size=1024,
        num_layers=5,
        act_type="silu",
        norm_type="layer",
        scale=1.0,
        dist_type="categorical",
        uniform_mix=0.01,
    )
    value_head_kwargs: Dict = FrozenDict(
        hid_size=1024,
        num_layers=5,
        act_type="silu",
        norm_type="layer",
        scale=0.0,
        dist_type="discrete",
        low=-20.0,
        high=20.0,
        trans_type="symlog",
    )
    normalizer_kwargs: Dict = FrozenDict(
        decay=0.99,
        max_scale=1.0,
        q_low=5.0,
        q_high=95.0,
    )
    # Other config fields for your heads or normalizer if needed.

    def setup(self):
        # Submodules define shapes but do NOT hold trainable params here.

        # Action head
        self.action_head = MLPHead((self.num_actions,), **self.action_head_kwargs)

        # Value head
        self.value_head = MLPHead((), **self.value_head_kwargs)

        # Slow value head
        self.slow_value_head = MLPHead((), **self.value_head_kwargs)

        # Normalizer
        self.normalizer = Normalizer(**self.normalizer_kwargs)

    def act(
        self,
        rng_key: jnp.ndarray,
        latent: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Samples an action from the action_head distribution using policy_params.
        Returns action.
        """
        dist = self.action_head(latent)
        action = dist.sample(seed=rng_key)
        return action

    def get_value(
        self,
        latent: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Returns the mean of the current (fast) value function.
        """
        dist = self.value_head(latent)
        return dist.mean()

    def get_slow_value(
        self,
        latent: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Returns the mean of the slow target value function.
        """
        dist = self.slow_value_head(latent)
        return dist.mean()


# ------------------------------------------------------------------------------
#  2) WorldModel with no loss-related code, just forward ops (encoder, rssm, decoder, etc.)
# ------------------------------------------------------------------------------

class RSSM(Module):
    """
    RSSM for forward passes.
    Remove or simplify any obs_step/img_step logic that used to handle states,
    because we'll handle them in the Agent now.
    """
    deter_size: int
    stoch_size: int
    num_classes: int
    # ... other config

    def setup(self):
        # e.g. define MLPs / GRUs / Dense layers
        # but do not store any trainable parameters directly here.
        self.gru_x_linear = ...
        self.gru_h_linear = ...
        # etc.

    def prior_forward(
        self,
        deter: jnp.ndarray,
        stoch: jnp.ndarray,
        action: jnp.ndarray,
        rng_key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        One step of 'img_step' style forward:
          - combine stoch + action
          - run through GRU
          - produce new deter, logit, stoch
        Return (new_deter, new_logit, new_stoch).
        """
        raise NotImplementedError("Define your prior forward pass logic here.")

    def posterior_forward(
        self,
        deter: jnp.ndarray,
        encoded: jnp.ndarray,
        rng_key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        One step of 'obs_step' style forward for the posterior distribution.
        Return (logit, stoch).
        """
        raise NotImplementedError("Define your posterior forward pass logic here.")


class WorldModel(Module):
    """
    Bare-bones forward module: no losses, just call the submodules.
    """
    obs_shape: Sequence[int]
    num_actions: int
    # ... any other config

    def setup(self):
        self.encoder = ...  # e.g. CNNEncoder(...)
        self.rssm = RSSM(...)  # with your config
        self.decoder = ...  # e.g. CNNDecoder(...)
        self.reward_head = ...
        self.cont_head = ...

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(obs)

    def decode(self, latent: jnp.ndarray):
        return self.decoder(latent)  # returns a distribution

    def forward_prior(
        self,
        deter: jnp.ndarray,
        stoch: jnp.ndarray,
        action: jnp.ndarray,
        rng_key: jnp.ndarray
    ):
        return self.rssm.prior_forward(deter, stoch, action, rng_key)

    def forward_posterior(
        self,
        deter: jnp.ndarray,
        encoded: jnp.ndarray,
        rng_key: jnp.ndarray
    ):
        return self.rssm.posterior_forward(deter, encoded, rng_key)

    def reward_forward(self, latent: jnp.ndarray):
        return self.reward_head(latent)

    def cont_forward(self, latent: jnp.ndarray):
        return self.cont_head(latent)


# ------------------------------------------------------------------------------
#  3) Agent class that centralizes all “observe”, “act”, and “imagine” methods
# ------------------------------------------------------------------------------

@struct.dataclass
class RSSMState:
    """Example container for RSSM hidden variables."""
    deter: jnp.ndarray
    logit: jnp.ndarray
    stoch: jnp.ndarray

@struct.dataclass
class AgentState:
    """
    State the Agent tracks across time:
      - RSSM hidden state
      - last action
    """
    rssm_state: RSSMState
    prev_action: jnp.ndarray


class Agent(Module):
    """
    Wraps the WorldModel + Policy.
    Now we hold the logic for observe, act, imagine, etc. in here.
    """
    obs_shape: Sequence[int]
    num_actions: int
    img_horizon: int = 15

    def setup(self):
        self.model = WorldModel(self.obs_shape, self.num_actions)
        self.policy = Policy(self.num_actions)
        # Possibly define an initial RSSM param for deter, or however you want to do it.

    def initial_agent_state(self, batch_size: int) -> AgentState:
        """
        Creates a fresh AgentState with default or learned initial RSSM state,
        zero action, and default policy state.
        """
        # For example, you might define a param for initial_deter etc.
        # We do not show that detail here. Suppose they are zeros:
        rssm_state = RSSMState(
            deter=jnp.zeros((batch_size, ...)),
            logit=jnp.zeros((batch_size, ...)),
            stoch=jnp.zeros((batch_size, ...)),
        )
        prev_action = jnp.zeros((batch_size, self.num_actions), jnp.float32)

        return AgentState(rssm_state, prev_action, policy_state)

    def observe(
        self,
        rng_key: jnp.ndarray,
        agent_state: AgentState,
        obs: jnp.ndarray,
        first: jnp.ndarray
    ) -> AgentState:
        """
        Example 'observe' logic: 
         1) encode obs
         2) if first=1, reset rssm_state
         3) prior forward to get new deter
         4) posterior forward to correct stoch from the encoded obs
         5) store results in new RSSMState
        """
        # Split RNG for prior/post
        rng_key_prior, rng_key_post = jax.random.split(rng_key, 2)

        # 1) encode
        encoded = self.model.encode(obs)

        # 2) if first=1 => reset
        condition = 1.0 - first.astype(jnp.float32)
        old = agent_state.rssm_state
        # (You might do your own param-based initial_deter here.)
        init_rssm = RSSMState(
            deter=jnp.zeros_like(old.deter),
            logit=jnp.zeros_like(old.logit),
            stoch=jnp.zeros_like(old.stoch),
        )
        # Mask them
        deter = condition * old.deter + (1 - condition) * init_rssm.deter
        logit = condition * old.logit + (1 - condition) * init_rssm.logit
        stoch = condition * old.stoch + (1 - condition) * init_rssm.stoch

        # 3) prior forward
        new_deter, new_logit_prior, new_stoch_prior = self.model.forward_prior(
            deter, stoch, agent_state.prev_action, rng_key_prior
        )

        # 4) posterior forward
        # the posterior reuses the new_deter but merges in obs encoding
        post_logit, post_stoch = self.model.forward_posterior(
            new_deter, encoded, rng_key_post
        )

        new_rssm_state = RSSMState(
            deter=new_deter,
            logit=post_logit,
            stoch=post_stoch
        )

        # 5) return updated AgentState with the new RSSM state
        return agent_state.replace(rssm_state=new_rssm_state)

    def act(
        self,
        rng_key: jnp.ndarray,
        agent_state: AgentState,
    ) -> Tuple[jnp.ndarray, AgentState]:
        """
        Samples an action from the policy based on the current posterior (rssm_state).
        Returns (action, updated AgentState).
        """
        # Flatten the stoch in agent_state.rssm_state as needed:
        deter = agent_state.rssm_state.deter
        stoch_flat = agent_state.rssm_state.stoch.reshape(deter.shape[0], -1)
        latent = jnp.concatenate([deter, stoch_flat], axis=-1)

        # Get an action from the policy
        action = self.policy.act(
            rng_key,
            latent,
        )

        # Store the new action as prev_action
        new_agent_state = agent_state.replace(
            prev_action=action,
        )
        return action, new_agent_state

    def imagine(
        self,
        rng_keys: jnp.ndarray,  # shape [T, ...]
        agent_state: AgentState,
    ) -> Dict[str, jnp.ndarray]:
        """
        Example 'imagine' logic to produce a rollout of length T = rng_keys.shape[0].
        We do a 'prior forward' step each time, sample an action, etc.
        """
        def flatten(x: jnp.ndarray) -> jnp.ndarray:
            return x.reshape(-1, *x.shape[2:])

        # Flatten state if needed. Below is an example if you had [B, T] shapes to start with.
        # We'll just show the logic for running a scan over time dimension:
        carry_init = agent_state

        def scan_fn(carry, rng):
            old_state = carry
            # 1) prior forward from old_state
            deter = old_state.rssm_state.deter
            stoch = old_state.rssm_state.stoch
            action = old_state.prev_action

            new_deter, logit, stoch_new = self.model.forward_prior(deter, stoch, action, rng)
            new_rssm = RSSMState(deter=new_deter, logit=logit, stoch=stoch_new)

            # 2) sample new action from policy
            stoch_flat = stoch_new.reshape(stoch_new.shape[0], -1)
            latent = jnp.concatenate([new_deter, stoch_flat], axis=-1)
            rng_act = jax.random.fold_in(rng, 0)  # or some other split
            act = self.policy.act(rng_act, latent)

            new_state = old_state.replace(
                rssm_state=new_rssm,
                prev_action=act,
            )

            # Return any info for logging
            return new_state, (latent, act)

        final_state, (latents, actions) = jax.lax.scan(scan_fn, carry_init, rng_keys)
        return {
            "latents": latents,
            "actions": actions,
        }
