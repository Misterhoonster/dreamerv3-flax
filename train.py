import argparse
from functools import partial
from typing import Dict, Sequence
import wandb

import numpy as np
import jax
import jax.numpy as jnp

# from dreamerv3_flax.env import CrafterEnv, VecCrafterEnv, TASKS
# from dreamerv3_flax.jax_agent import JAXAgent
from dreamerv3_hoon.async_vector_env import AsyncVectorEnv  

from dreamerv3_hoon.jax_agent import JAXAgent
from dreamerv3_hoon.buffer import ReplayBuffer
from dreamerv3_hoon.env import CatchEnv, VecCatchEnv


# def get_eval_metric(achievements: Sequence[Dict]) -> float:
#     achievements = [list(achievement.values()) for achievement in achievements]
#     success_rate = 100 * (np.array(achievements) > 0).mean(axis=0)
#     score = np.exp(np.mean(np.log(1 + success_rate))) - 1
#     eval_metric = {
#         "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
#         "score": score,
#     }
#     return eval_metric


def main(args):
    # Logger
    project = "dreamer-catch"
    group = f"{args.exp_name}"
    if args.timestamp:
        group += f"-{args.timestamp}"
    name = f"s{args.seed}"
    logger = wandb.init(project=project, group=group, name=name)

    # Seed
    np.random.seed(args.seed)

    # Environment
    # env_fns = [partial(CatchEnv, seed=args.seed)]
    # env = VecCatchEnv(AsyncVectorEnv(env_fns))

    env = VecCatchEnv(num_envs=8, seed=args.seed)

    # Buffer
    buffer = ReplayBuffer(env, batch_size=16, num_steps=64)

    # Agent
    agent = JAXAgent(env, seed=args.seed)
    state = agent.initial_state(1)

    # Reset
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, env.num_envs)
    actions = jax.vmap(env.action_space.sample)(keys)
    print("initial actions shape:", actions.shape)
    obs, rewards, dones, firsts, infos = env.step(actions)

    # Train
    for step in range(5000):
        actions, state = agent.act(obs, firsts, state)
        buffer.add(obs, actions, rewards, dones, firsts)

        actions = np.argmax(actions, axis=-1)
        obs, rewards, dones, firsts, infos = env.step(actions)

        # Get metrics for done episodes
        done_returns = infos["episode_return"][dones]
        done_lengths = infos["episode_length"][dones]
        
        # Only log if any episodes finished
        if jnp.any(dones):
            # Log each finished episode
            for ret, length in zip(done_returns, done_lengths):
                rollout_metric = {
                    "episode_return": ret,
                    "episode_length": length,
                }
                logger.log(rollout_metric, step)
        # for done, info in zip(dones, infos):
        #     if done:
        #         print("INFO:", info)
        #         rollout_metric = {
        #             "episode_return": info["episode_return"],
        #             "episode_length": info["episode_length"],
        #         }
        #         logger.log(rollout_metric, step)
                # eval_metric = get_eval_metric(info["achievements"])
                # logger.log(eval_metric, step)

        if step >= 1024 and step % 2 == 0:
            data = buffer.sample()
            _, train_metric = agent.train(data)
            if step % 100 == 0:
                logger.log(train_metric, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    main(args)
