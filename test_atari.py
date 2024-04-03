from src.envs.pettingzoo_env import *
from src.data import *
from src.models.az_resnet import *

test_env = space_invaders_v2.env()
test_env.reset()
print(test_env.agents)

wrapped_env = from_pettingzoo(test_env)

tbuf = TrajectoryBuffer(5, wrapped_env)

afn = AZResNet(wrapped_env.OBS_SHAPE, wrapped_env.ACTION_DIM)

tbuf = gen_batch_traj_buffer(tbuf, wrapped_env, afn, 4, 2)
