import copy

import supersuit as ss
from torch import optim
import wandb

from src.envs.pettingzoo_env import *
from pettingzoo.atari import pong_v3
from src.data import *
from src.models.az_resnet import *
from src.tb import train


max_cycles = 50


class AttrDict(dict):
    def __getattr__(self, k):
        return self[k]


env = pong_v3.env(max_cycles=max_cycles, obs_type="grayscale_image")

obs_size = (84, 84)
frame_stack = 4
env = ss.max_observation_v0(env, 2)
env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
env = ss.frame_skip_v0(env, 4)
env = ss.resize_v1(env, x_size=obs_size[0], y_size=obs_size[1])
env = ss.frame_stack_v1(env, frame_stack)

wrapped_env = from_pettingzoo(env, max_cycles=max_cycles, obs_size=(4, *obs_size), terminate_on_reward=True)

# tbuf = TrajectoryBuffer(5, wrapped_env)
afn = AZResNet((frame_stack+1, *obs_size), wrapped_env.ACTION_DIM)
# tbuf = gen_batch_traj_buffer(tbuf, wrapped_env, afn, 4, 2)

optimizer = optim.Adam([{"params": afn.parameters(), "lr": 1e-3}, {"params": afn.log_Z_0, "lr": 5e-2, },
                        {"params": afn.log_Z_1, "lr": 5e-2, }])
buffer = TrajectoryBuffer(2, wrapped_env)
train_cfg = AttrDict({
    "batch_size": 2,
    "total_steps": 20_000,
    "eval_every": 30000,
    "buffer_batch_size": 2,
    "num_initial_traj": 2,
    "num_regen_traj": 2,
    "regen_every": 2,
    "ckpt_dir": "~/scratch/checkpoints-Pong"
})

train_cfg.__getattribute__ = lambda self, x: self[x]

# wandb.init(project="AFN", tags=["TankTrouble"])

train(wrapped_env, afn, optimizer, buffer, train_cfg)


