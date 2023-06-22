import sys
from typing import Any, SupportsFloat
from typing_extensions import Self

import numpy as np
from gymnasium.core import ObsType, RenderFrame, ActType
from gymnasium.error import ResetNeeded
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium import ActionWrapper
from rlberry.envs import FiniteMDP
import matplotlib.pyplot as plt
import matplotlib as mpl


class GridEnvActionWrapper(ActionWrapper):

    ACTIONS_MAPPING = {
        'up': 0,
        'right': 1,
        'down': 2,
        'left': 3,
        'unchanged': 4,
    }

    def action(self, action: ActType) -> ActType:
        if isinstance(action, int):
            return action - 1
        elif isinstance(action, str):
            return self.ACTIONS_MAPPING[action]


class _GridEnv(FiniteMDP):

    metadata: dict[str, Any] = {'render_modes': ['human', 'ascii']}
    MOVES: list[tuple[str, tuple[int, int]]] = [
        ('up', (-1, 0)),
        ('right', (0, 1)),
        ('down', (1, 0)),
        ('left', (0, -1)),
        ('unchanged', (0, 0)),
    ]

    def __init__(
            self: Self, 
            size: int,
            gamma: float=0.9,
            reward_goal: float=1.0,
            reward_other: float=0.0,
            reward_forbidden: float=-1.0,
            reward_boundary: float=-1.0,
            state_start=None,
            state_goal=None,
            states_forbidden=None,
            render_mode: str | None = None,
            max_iter: int = 100,
            render_fps: float = 10.0,
        ) -> None:
        self.size = size
        self.gamma = gamma
        self.reward_goal = reward_goal
        self.reward_other = reward_other
        self.reward_forbidden = reward_forbidden
        self.reward_boundary = reward_boundary
        self.state_start = state_start
        self.state_goal = state_goal
        self.states_forbidden = states_forbidden
        self.render_mode = render_mode
        self.max_iter = max_iter
        self.render_fps = render_fps
        self._init_mpd()
        self._reseted = False
    
    def _state_to_obs(self: Self, state: tuple[int, int]) -> ObsType:
        return state[0] * self.size + state[1]
    
    def _obs_to_state(self: Self, obs: ObsType) -> tuple[int, int]:
        return np.array([obs // self.size, obs % self.size])
    
    def _in_bounds(self: Self, state: tuple[int, int]) -> bool:
        y, x = state
        return 0 <= x <= self.size - 1 and 0 <= y <= self.size - 1

    def _next_state(self: Self, state: ObsType, action: ActType) -> bool:
        dstate = np.array(self.MOVES[action][1])
        next_state = state + dstate
        return next_state if self._in_bounds(next_state) else state
    
    def _reset(self: Self) -> None:
        self._n_iter = 0
        self._n_iter_rendered = 0
        self._transitions = []
        self._reseted = True

    def _init_mpd(self: Self) -> None:

        num_states, num_actions =  self.size ** 2, len(self.MOVES)
        
        # Mean rewards array
        R = np.zeros((num_states, num_actions))
        
        self.reward_other_ = self.reward_other
        R += self.reward_other_
        
        self.state_start_ = np.array(self.state_start if self.state_start is not None else (0, 0))
        initial_state_distribution = self._state_to_obs(self.state_start_)
        
        self.state_goal_ = np.array(self.state_goal if self.state_goal is not None else (self.size - 1, self.size -1))
        self.reward_goal_ = self.reward_goal
        for obs in range(num_states):
            for action in range(num_actions):
                next_state = self._next_state(self._obs_to_state(obs), action)
                if np.array_equal(next_state, self.state_goal_):
                    R[obs, action] = self.reward_goal_

        self.states_forbidden_ = [np.array(state) for state in self.states_forbidden] if self.states_forbidden is not None else np.array([])
        self.reward_forbidden_ = self.reward_forbidden
        for obs in range(num_states):
            for action in range(num_actions):
                next_state = self._next_state(self._obs_to_state(obs), action)
                if any(np.array_equal(next_state, state) for state in self.states_forbidden_):
                    R[obs, action] = self.reward_forbidden_

        self.reward_boundary_ = self.reward_boundary
        for obs in range(num_states):
            for action in range(num_actions):
                next_state = self._next_state(self._obs_to_state(obs), action)
                state = self._obs_to_state(obs)
                if np.array_equal(next_state, state) and action != num_actions - 1:
                    R[obs, action] = self.reward_boundary_
        
        # Transition probabilities array
        P = np.zeros((num_states, num_actions, num_states))
        for obs, probs in enumerate(P):
            for action in range(num_actions):
                next_obs = self._state_to_obs(self._next_state(self._obs_to_state(obs), action))
                probs[action, next_obs] = 1.0
        
        FiniteMDP.__init__(self, R, P, initial_state_distribution)

    def _render_backround(self: Self) -> None:
        plt.ion()
        self._ax = plt.subplots()[1]
        cmap = mpl.colors.ListedColormap(['white', 'cyan', 'orange'])
        data = np.zeros((self.size, self.size))
        data[self.state_goal_[0], self.state_goal_[1]] = 1
        for state in self.states_forbidden_:
            data[state[0], state[1]] = 2
        self._ax.pcolormesh(data, cmap=cmap)
        self._ax.grid(axis='both', color='k', linewidth=1.0) 
        self._ax.set_xticks(np.arange(0, self.size, 1))
        self._ax.set_yticks(np.arange(0, self.size, 1))
        self._ax.tick_params(
            bottom=False, 
            top=False, 
            left=False, 
            right=False, 
            labelbottom=False, 
            labelleft=False
        )
        self._ax.invert_yaxis()
        self._ax.set_title('Step: 0, Discounted return: 0')
        patch_start = mpl.patches.Circle(
            (self.state_start_[1] + .5, self.state_start_[0] + .5), 
            0.03, 
            facecolor='black', 
            fill=True, 
            zorder=100,
        )
        self._ax.add_patch(patch_start)
        for obs in range(0, self.unwrapped.size ** 2):
            y, x = self.unwrapped._obs_to_state(obs)
            self._ax.text(x + 0.5, y + 0.5, f'$S_{obs + 1}$', fontsize=14)

    def _render_state(self: Self, previous_obs: ObsType, obs: ObsType) -> None:
        y_start, x_start = self._obs_to_state(previous_obs)
        y_end, x_end = self._obs_to_state(obs)
        if x_start != x_end or y_start != y_end:
            self._ax.plot((x_start + .5, x_end + .5), (y_start + .5, y_end + .5), 'r', linestyle='dashed')
            self._ax.arrow(x_end + .5, y_end + .5, 0.01 * (x_end - x_start), 0.01 * (y_end - y_start), shape='full', lw=0, length_includes_head=True, head_width=.05, color='red')
        else:
            patch = mpl.patches.Circle(
                (x_start + .5, y_start + .5), 
                0.05, 
                color='red',
                linestyle='dashed',
                fill=False,
            )
            self._ax.add_patch(patch)

    @property
    def discounted_returns_(self: Self) -> list[float]:
        return [reward * self.gamma ** step for step, (*_, reward, _) in enumerate(self._transitions)]

    def reset(self: Self, *, seed: int | None = None, options: dict[str, Any] | None = None,) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment to the initial state."""
        self._reset()
        if hasattr(self, '_ax'):
            self._render_backround()
        return super().reset(seed, options)

    def step(self: Self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._reseted is False:
            raise ResetNeeded('Reset the environment before calling the `step` method.')
        previous_obs = self.state
        obs, reward, *_, info = super().step(action)
        self._n_iter += 1
        truncated = self._n_iter >= self.max_iter
        self._transitions.append((previous_obs, obs, reward, action))
        self.render()
        return obs, reward, False, truncated, info

    def render(self: Self)-> RenderFrame | list[RenderFrame] | None:
        if self._reseted is False:
            raise ResetNeeded('Reset the environment before calling the `render` method.')
        render_modes = self.metadata['render_modes']
        if self.render_mode is None:
            return
        elif self.render_mode not in render_modes:
            raise ValueError(f'The supported render modes are {", ".join(render_modes)}. Got {self.render_mode} instead.')
        elif self.render_mode == 'human':
            if not hasattr(self, '_ax'):
                self._render_backround()
            for step, (previous_obs, obs, _, _) in enumerate(self._transitions):
                if self._n_iter_rendered < self._n_iter:
                    if self.render_fps > 0.0:
                        plt.pause(1 / self.render_fps)
                    self._n_iter_rendered += 1
                self._render_state(previous_obs, obs)
                self._ax.set_title(f"Step: {step + 1}, Discounted return: {sum(self.discounted_returns_)}")
        if self.render_mode == 'ascii':
            trajectory = f'||s_0={self._transitions[0][0]}, r_0=0.0||  '
            for step, (_, obs_end, reward, action) in enumerate(self._transitions):
                trajectory += f'--a_{step}={action}--> ||s_{step + 1}={obs_end}, r_{step + 1}={reward}||  '
            print(trajectory.strip())
    
    def plot(self: Self, kind: str):
        plt.ion()
        if kind == 'observation_space':
            ax = plt.subplots()[1]
            cmap = mpl.colors.ListedColormap(['white', 'cyan', 'orange'])
            data = np.zeros((self.unwrapped.size, self.unwrapped.size))
            data[self.unwrapped.state_goal_[0], self.unwrapped.state_goal_[1]] = 1
            for state in self.unwrapped.states_forbidden_:
                data[state[0], state[1]] = 2
            ax.pcolormesh(data, cmap=cmap)
            ax.grid(axis='both', color='k', linewidth=1.0) 
            ax.set_xticks(np.arange(0, self.unwrapped.size, 1))
            ax.set_yticks(np.arange(0, self.unwrapped.size, 1))
            ax.tick_params(
                bottom=False, 
                top=False, 
                left=False, 
                right=False, 
                labelbottom=False, 
                labelleft=False
            )
            ax.invert_yaxis()
            ax.set_title('Observation space')
            for obs in range(0, self.unwrapped.size ** 2):
                y, x = self.unwrapped._obs_to_state(obs)
                ax.text(x + 0.5, y + 0.5, f'$S_{obs + 1}$')
        elif kind == 'action_space':
            ax = plt.subplots()[1]
            ax.axis('off')
            rectangle = plt.Rectangle((0.3, 0.3), 0.4, 0.4, fill=False)
            ax.add_patch(rectangle)
            circle = plt.Circle(
                (.5, .5), 
                0.03,
                color='green', 
                fill=False, 
            )
            ax.add_patch(circle)
            ax.arrow(.6, .5, .3, .0, shape='full', lw=0, length_includes_head=True, head_width=.03, color='green')
            ax.text(0.92, 0.49, '$a_2$')
            ax.arrow(.4, .5, -.3, .0, shape='full', lw=0, length_includes_head=True, head_width=.03, color='green')
            ax.text(0.05, 0.49, '$a_4$')
            ax.arrow(.5, .6, .0, .3, shape='full', lw=0, length_includes_head=True, head_width=.03, color='green')
            ax.text(0.49, 0.92, '$a_1$')
            ax.arrow(.5, .4, 0, -.3, shape='full', lw=0, length_includes_head=True, head_width=.03, color='green')
            ax.text(0.49, 0.05, '$a_3$')
            ax.text(0.53, 0.43, '$a_5$')
            ax.set_title('Action space')

    def close(self: Self, *args):
        """
        Close the environment.
        """
        plt.close()
        sys.exit()


def GridEnv(
        size: int,
        gamma: float=0.9,
        reward_goal: float=1.0,
        reward_other: float=0.0,
        reward_forbidden: float=-1.0,
        reward_boundary: float=-1.0,
        state_start=None,
        state_goal=None,
        states_forbidden=None,
        render_mode: str | None = None,
        max_iter: int = 100,
        render_fps: float = 10.0,
    ) -> None:
    env = _GridEnv(
        size=size,
        gamma=gamma,
        reward_goal=reward_goal,
        reward_other=reward_other,
        reward_forbidden=reward_forbidden,
        reward_boundary=reward_boundary,
        state_start=state_start,
        state_goal=state_goal,
        states_forbidden=states_forbidden,
        render_mode=render_mode,
        max_iter=max_iter,
        render_fps=render_fps,
    )
    env = TransformObservation(env, lambda obs: obs + 1)
    env = GridEnvActionWrapper(env)
    return env 


def main() -> None:
    env = GridEnv(
        size=3,
        states_forbidden=[(1, 2), (2, 0)],
        render_mode='human',
        render_fps=0.1
    )
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, _, truncated, info = env.step(action)
        if truncated:
            break
    
if __name__ == "__main__":
    main()
