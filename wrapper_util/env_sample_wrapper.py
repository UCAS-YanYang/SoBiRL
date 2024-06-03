"""Environment wrappers for collecting rollouts."""

from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from imitation.data import types,rollout
from sample_util import cus_rollout


class BufferingWrapper(VecEnvWrapper):
    """Saves transitions of underlying VecEnv.

    Retrieve saved transitions using `pop_transitions()`.
    """

    error_on_premature_event: bool
    _trajectories: List[types.TrajectoryWithRew]
    _ep_lens: List[int]
    _init_reset: bool
    _traj_accum: Optional[cus_rollout.TrajectoryAccumulator]
    _timesteps: Optional[npt.NDArray[np.int_]]
    n_transitions: Optional[int]

    def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
        """Builds BufferingWrapper.

        Args:
            venv: The wrapped VecEnv.
            error_on_premature_reset: Error if `reset()` is called on this wrapper
                and there are saved samples that haven't yet been accessed.
        """
        super().__init__(venv)
        self.error_on_premature_reset = error_on_premature_reset
        self._trajectories = []
        self._ep_lens = []
        self._init_reset = False
        self._traj_accum = None
        self._saved_acts = None
        self._timesteps = None
        self.n_transitions = None

    def reset(self, **kwargs):
        if (
            self._init_reset
            and self.error_on_premature_reset
            and self.n_transitions > 0
        ):  # noqa: E127
            raise RuntimeError("BufferingWrapper reset() before samples were accessed")
        self._init_reset = True
        self.n_transitions = 0
        obs = self.venv.reset(**kwargs)
        self._traj_accum = cus_rollout.TrajectoryAccumulator()
        obs = types.maybe_wrap_in_dictobs(obs)
        for i, ob in enumerate(obs):
            self._traj_accum.add_step({"obs": ob}, key=i)
        self._timesteps = np.zeros((len(obs),), dtype=int)
        obs = types.maybe_unwrap_dictobs(obs)
        return obs

    def step_async(self, actions):
        assert self._init_reset
        assert self._saved_acts is None
        self.venv.step_async(actions)
        self._saved_acts = actions

    def step_wait(self):
        assert self._init_reset
        assert self._saved_acts is not None
        acts, self._saved_acts = self._saved_acts, None
        obs, rews, dones, infos = self.venv.step_wait()

        self.n_transitions += self.num_envs
        self._timesteps += 1
        ep_lens = self._timesteps[dones]
        if len(ep_lens) > 0:
            self._ep_lens += list(ep_lens)
        self._timesteps[dones] = 0

        finished_trajs = self._traj_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        self._trajectories.extend(finished_trajs)

        return obs, rews, dones, infos

    def _finish_partial_trajectories(self) -> Sequence[types.TrajectoryWithRew]:
        """Finishes and returns partial trajectories in `self._traj_accum`."""
        assert self._traj_accum is not None
        trajs = []
        for i in range(self.num_envs):
            # Check that we have any transitions at all.
            # The number of "transitions" or "timesteps" stored for the ith
            # environment is the number of step dicts stored in
            # `partial_trajectories[i]` minus one. We need to offset by one because
            # the first step dict is comes from `reset()`, not from `step()`.
            n_transitions = len(self._traj_accum.partial_trajectories[i]) - 1
            assert n_transitions >= 0, "Invalid TrajectoryAccumulator state"
            if n_transitions >= 1:
                traj = self._traj_accum.finish_trajectory(i, terminal=False)
                trajs.append(traj)

                # Reinitialize a partial trajectory starting with the final observation.
                self._traj_accum.add_step({"obs": traj.obs[-1]}, key=i)
        return trajs

    def pop_finished_trajectories(
        self,
    ) -> Tuple[Sequence[types.TrajectoryWithRew], Sequence[int]]:
        """Pops recorded complete trajectories `trajs` and episode lengths `ep_lens`.

        Returns:
            A tuple `(trajs, ep_lens)` where `trajs` is a sequence of trajectories
            including the terminal state (but possibly missing initial states, if
            `pop_trajectories` was previously called) and `ep_lens` is a sequence
            of episode lengths. Note the episode length will be longer than the
            trajectory length when the trajectory misses initial states.
        """
        trajectories = self._trajectories
        ep_lens = self._ep_lens
        self._trajectories = []
        self._ep_lens = []
        self.n_transitions = 0
        return trajectories, ep_lens

    def pop_trajectories(
        self,
    ) -> Tuple[Sequence[types.TrajectoryWithRew], Sequence[int]]:
        """Pops recorded trajectories `trajs` and episode lengths `ep_lens`.

        Returns:
            A tuple `(trajs, ep_lens)`. `trajs` is a sequence of trajectory fragments,
            consisting of data collected after the last call to `pop_trajectories`.
            They may miss initial states (if `pop_trajectories` previously returned
            a fragment for that episode), and terminal states (if the episode has
            yet to complete). `ep_lens` is the total length of completed episodes.
        """
        if self.n_transitions == 0:
            return [], []
        partial_trajs = self._finish_partial_trajectories()
        self._trajectories.extend(partial_trajs)
        return self.pop_finished_trajectories()

    def pop_transitions(self) -> types.TransitionsWithRew:
        """Pops recorded transitions, returning them as an instance of Transitions.

        Returns:
            All transitions recorded since the last call.

        Raises:
            RuntimeError: empty (no transitions recorded since last pop).
        """
        if self.n_transitions == 0:
            # It would be better to return an empty `Transitions`, but we would need
            # to get the non-zero dimensions of every np.ndarray attribute correct to
            # avoid downstream errors. This is easier and sufficient for now.
            raise RuntimeError("Called pop_transitions on an empty BufferingWrapper")
        # make a copy for the assert later
        n_transitions = self.n_transitions
        trajectories, _ = self.pop_trajectories()
        transitions = rollout.flatten_trajectories_with_rew(trajectories)
        assert len(transitions.obs) == n_transitions
        return transitions


class RolloutInfoWrapper(gym.Wrapper):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the NumPy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env: gym.Env):
        """Builds RolloutInfoWrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs = None
        self._rews = None

    def reset(self, **kwargs):
        new_obs, info = super().reset(**kwargs)
        self._obs = [types.maybe_wrap_in_dictobs(new_obs)]
        self._rews = []
        return new_obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._obs.append(types.maybe_wrap_in_dictobs(obs))
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": types.stack_maybe_dictobs(self._obs),
                "rews": np.stack(self._rews),
            }
        return obs, rew, terminated, truncated, info



from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import gymnasium as gym
import numpy as np

# Note: we redefine the type vars from gymnasium.core here, because pytype does not
# recognize them as valid type vars if we import them from gymnasium.core.
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class AutoResetWrapper(
    gym.Wrapper,
    Generic[WrapperObsType, WrapperActType, ObsType, ActType],
):
    """Hides terminated truncated and auto-resets at the end of each episode.

    Depending on the flag 'discard_terminal_observation', either discards the terminal
    observation or pads with an additional 'reset transition'. The former is the default
    behavior.
    In the latter case, the action taken during the 'reset transition' will not have an
    effect, the reward will be constant (set by the wrapper argument `reset_reward`,
    which has default value 0.0), and info an empty dictionary.
    """

    def __init__(self, env, discard_terminal_observation=True, reset_reward=0.0):
        """Builds the wrapper.

        Args:
            env: The environment to wrap.
            discard_terminal_observation: Defaults to True. If True, the terminal
                observation is discarded and the environment is reset immediately. The
                returned observation will then be the start of the next episode. The
                overridden observation is stored in `info["terminal_observation"]`.
                If False, the terminal observation is returned and the environment is
                reset in the next step.
            reset_reward: The reward to return for the reset transition. Defaults to
                0.0.
        """
        super().__init__(env)
        self.discard_terminal_observation = discard_terminal_observation
        self.reset_reward = reset_reward
        self.previous_done = False  # Whether the previous step returned done=True.

    def step(
        self,
        action: WrapperActType,
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """When terminated or truncated, resets the environment.

        Always returns False for terminated and truncated.

        Depending on whether we are discarding the terminal observation,
        either resets the environment and discards,
        or returns the terminal observation, and then uses the next step to reset the
        environment, after which steps will be performed as normal.
        """
        if self.discard_terminal_observation:
            return self._step_discard(action)
        else:
            return self._step_pad(action)

    def _step_pad(
        self,
        action: WrapperActType,
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """When terminated or truncated, resets the environment.

        Always returns False for terminated and truncated.

        The agent will then usually be asked to perform an action based on
        the terminal observation. In the next step, this final action will be ignored
        to instead reset the environment and return the initial observation of the new
        episode.

        Some potential caveats:
        - The underlying environment will perform fewer steps than the wrapped
          environment.
        - The number of steps the agent performs and the number of steps recorded in the
          underlying environment will not match, which could cause issues if these are
          assumed to be the same.
        """
        if self.previous_done:
            self.previous_done = False
            reset_obs, reset_info_dict = self.env.reset()
            info = {"reset_info_dict": reset_info_dict}
            # This transition will only reset the environment, the action is ignored.
            return reset_obs, self.reset_reward, False, False, info

        obs, rew, terminated, truncated, info = self.env.step(action)
        self.previous_done = terminated or truncated
        return obs, rew, terminated, truncated, info

    def _step_discard(
        self,
        action: WrapperActType,
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """When terminated or truncated, return False for both and automatically reset.

        When an automatic reset happens, the observation from reset is returned,
        and the overridden observation is stored in
        `info["terminal_observation"]`.
        """
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["terminal_observation"] = obs
            obs, reset_info_dict = self.env.reset()
            info["reset_info_dict"] = reset_info_dict
        return obs, rew, terminated, truncated, info
