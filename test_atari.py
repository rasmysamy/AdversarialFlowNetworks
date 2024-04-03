import copy

from src.envs.pettingzoo_env import *
from src.data import *
from src.models.az_resnet import *

max_cycles = 50

test_env = space_invaders_v2.env(max_cycles=max_cycles)
test_env.reset()
print(test_env.agents)

test_env.step(1)


wrapped_env = from_pettingzoo(test_env, max_cycles=max_cycles)

assert test_env == wrapped_env._env

tbuf = TrajectoryBuffer(5, wrapped_env)

afn = AZResNet((4, 210, 160), wrapped_env.ACTION_DIM)

tbuf = gen_batch_traj_buffer(tbuf, wrapped_env, afn, 4, 2)
