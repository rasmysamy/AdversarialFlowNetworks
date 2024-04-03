import copy

from src.envs.pettingzoo_env import *
from src.data import *
from src.models.az_resnet import *

max_cycles = 1000



test_env = space_invaders_v2.env(max_cycles=max_cycles, obs_type="grayscale_image")
test_env.reset()
print(test_env.agents)

env = test_env

obs_size = (84, 84)
frame_stack = 4
env = ss.max_observation_v0(env, 2)
env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
env = ss.frame_skip_v0(env, 4)
env = ss.resize_v1(env, x_size=obs_size[0], y_size=obs_size[1])
env = ss.frame_stack_v1(env, frame_stack)
env.reset()


wrapped_env = from_pettingzoo(env, max_cycles=max_cycles, obs_size=(4, *obs_size))
tbuf = TrajectoryBuffer(5, wrapped_env)
afn = AZResNet((frame_stack+1, *obs_size), wrapped_env.ACTION_DIM)
tbuf = gen_batch_traj_buffer(tbuf, wrapped_env, afn, 4, 2)
