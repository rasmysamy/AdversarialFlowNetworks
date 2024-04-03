import math
from typing import Type

import numpy as np
from pettingzoo import AECEnv

from pettingzoo.atari import space_invaders_v2

from src.envs.base_env import BaseEnv, Outcome, Player

test_env = space_invaders_v2.env()
test_env.reset()
print(test_env.agents)

class WrappedPettingZooEnv(BaseEnv):
    def __init__(self, pettingzoo_env: AECEnv, max_cycles=10_000, rescale_factor: int = 1, *args, **kwargs):
        self._env = pettingzoo_env
        self._env.reset()

        self.obs_transformer = lambda x: x

        def winning_default(rewards_1: np.ndarray, rewards_2: np.ndarray):
            if np.mean(rewards_1) == np.mean(rewards_2):
                return Outcome.DRAW
            return Outcome.WIN_P1 if np.mean(rewards_1) > np.mean(rewards_2) else Outcome.WIN_P2

        self.evaluate_winner = winning_default

        agent_names = self._env.agents

        assert len(agent_names) == 2, "Only 2 player games are supported"

        # State vars
        self.board = np.zeros((int(210*rescale_factor), int(160*rescale_factor), 3))
        self.turns = 0
        self.curr_player = Player.ONE
        self.done = False
        self.Outcome = Outcome.NOT_DONE

        self.rewards = [np.array([]), np.array([])]

        # NEED TO BE SET BY EACH ENV
        name: "Wrapped Petting Zoo Environment"
        self.OBS_SHAPE: tuple[int, ...] = self._env.observation_space(agent_names[0]).shape
        self.NUM_EXTRA_INFO: int = 1
        self.ACTION_DIM: int = math.prod(self._env.action_space(agent_names[0]).shape)
        self.MAX_TRAJ_LEN: int = max_cycles * 2 + 1 # Number of possible turns + 1

        super().__init__(*args, **kwargs)

    def obs(self):
        return self.obs_transformer(self.board)

    def place_piece(self, action) -> None:
        self._env.step(action)
        self.curr_player.switch()
        self.turns += 1
        observation, reward, termination, truncation, info = self._env.last()
        self.board = observation
        if termination or truncation:
            self.done = True
            self.outcome = self.evaluate_outcome()
        self.rewards[self.curr_player] = np.append(self.rewards[self.curr_player], reward)

    def evaluate_outcome(self) -> Outcome:
        observation, reward, termination, truncation, info = self._env.last()
        if not termination:
            return Outcome.NOT_DONE
        return self.evaluate_winner(self.rewards[0], self.rewards[1])

    def get_extra_info(self) -> np.ndarray:
        return np.array([int(self.curr_player), self.turns])

    def get_masks(self) -> np.ndarray:
        return np.ones(self.ACTION_DIM)

    def reset(self) -> None:
        self._env.reset()
        observation, reward, termination, truncation, info = self._env.last()
        self.board = observation
        self.turns = 0
        self.curr_player = Player.ONE
        self.done = False
        self.outcome = Outcome.NOT_DONE
        self.rewards = [np.array([]), np.array([])]

    def render(self) -> None:
        pass

    def conv_obs(self) -> np.ndarray:
        extra_info = self.get_extra_info()
        obs = np.concatenate(
            [self.board] + [np.ones(self.CONV_SHAPE) * info for info in extra_info],
            axis=2,
        ).astype(np.float32)

        return obs

    def flat_obs(self) -> np.ndarray:
        obs = np.empty(self.FLAT_STATE_DIM, dtype=np.float32)
        obs[: self.OBS_DIM] = self.board.flatten()

        extra_info = self.get_extra_info()
        for i, info in enumerate(extra_info):
            obs[self.OBS_DIM + i] = info

        return obs



def from_pettingzoo(env: Type[AECEnv], *args, **kwargs):
    return WrappedPettingZooEnv(env, *args, **kwargs)