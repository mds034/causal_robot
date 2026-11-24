"""LivingRoom: a gridworld where a robot walks to a goal, avoiding a rug unless it is in a hurry."""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, NamedTuple, Optional

Pos = tuple[int, int]  # (row, col)


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MOVES = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}


class Obs(NamedTuple):
    """What an agent observes at each step. It never includes `hurry`."""

    pos: Pos
    on_rug: bool
    goal: Pos
    rug: tuple[int, int, int, int]  # top, left, bottom, right rows/cols of the rug (inclusive)


class LivingRoom:
    def __init__(
        self,
        size: int = 20,
        rug_shape: tuple[int, int] = (12, 14),
        rug_penalty: float = 5.0,
        hurry_penalty: float = 2.0,
        max_steps: int = 200,
        max_shove: int = 6,
        reward: str = 'dense',
    ):
        if reward not in ('dense', 'sparse'):
            raise ValueError(f'unknown reward mode {reward!r}')
        self.size = size
        self.reward_mode = reward  # 'dense': each step pays its cost; 'sparse': everything at the end

        # V
        self.start: Pos = (size // 2, 0)
        self.goal: Pos = (size // 2, size - 1)
        self.pos: Pos = self.start

        # U
        self.hurry: bool = None

        # create the rug as a (height x width) rectangle in the middle
        h, w = rug_shape
        top, left = (size - h) // 2, (size - w) // 2
        self.rug = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
        self.rug_bounds = (top, left, top + h - 1, left + w - 1)

        self.rug_penalty = rug_penalty
        self.hurry_penalty = hurry_penalty
        self.max_steps = max_steps
        self.max_shove = max_shove
        self.reset()

    def reset(self, hurry: Optional[bool] = None) -> Pos:
        self.hurry = random.random() < 0.2 if hurry is None else hurry
        self.pos = self.start
        self.steps = 0
        self.accrued = 0.0  # step rewards earned so far; paid out only when the episode ends
        self._cost = self._cost_to_go()
        return self.pos

    def step(self, action: Action) -> tuple[Pos, float, bool, dict]:
        dr, dc = MOVES[Action(action)]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        if self.in_bounds(nxt):
            self.pos = nxt

        self.steps += 1
        self.accrued += self.step_reward(self.pos)
        info = {'hurry': self.hurry}
        return self.pos, self.reward(), self.done, info

    def observe(self) -> Obs:
        # the agent's observation of the current state: position, whether it is on the rug, and the fixed layout
        return Obs(self.pos, self.pos in self.rug, self.goal, self.rug_bounds)

    def shove_problem(self, pos: Pos, direction: Action, distance: int) -> Optional[str]:
        # why a shove is not acceptable within the bounds of the environment, or None if it is
        if distance < 1:
            return 'too_short'
        if distance > self.max_shove:
            return 'too_long'
        dr, dc = MOVES[Action(direction)]
        if not self.in_bounds((pos[0] + dr * distance, pos[1] + dc * distance)):
            return 'out_of_room'
        return None

    def shove(self, direction: Action, distance: int) -> tuple[Pos, float, bool]:
        # a human correction: forces the robot `distance` steps in `direction` (stopping early at the goal).
        # Each forced step is an ordinary step, so it counts towards `steps` and accrues reward. A shove that is
        # too short, longer than `max_shove`, or would push the robot out of the room is rejected.
        problem = self.shove_problem(self.pos, direction, distance)
        if problem:
            raise ValueError(f'shove {Action(direction).name} {distance} from {self.pos} rejected: {problem}')
        total = 0.0
        for _ in range(distance):
            if self.done:
                break
            _, reward, _, _ = self.step(direction)
            total += reward
        return self.pos, total, self.done

    @property
    def done(self) -> bool:
        return self.pos == self.goal or self.steps >= self.max_steps

    def reward(self) -> float:
        # dense: the cost of the step just taken; sparse: nothing until the episode ends, then the whole return
        if self.reward_mode == 'dense':
            return self.step_reward(self.pos)
        return self.accrued if self.done else 0.0

    def step_reward(self, pos: Pos) -> float:
        # what stepping into `pos` adds to the episode's return (accrued, not paid out at once):
        # -1 per step; in a hurry every step costs extra, otherwise standing on the rug does.
        if self.hurry:
            return -1.0 - self.hurry_penalty
        return -1.0 - (self.rug_penalty if pos in self.rug else 0.0)

    def optimal_return(self) -> float:
        # best achievable return from the start under the current (hidden) `hurry`
        return -self._cost[self.start]

    def move(self, pos: Pos, action: Action) -> Optional[Pos]:
        # the cell one step from `pos`, or None if that would leave the room
        dr, dc = MOVES[Action(action)]
        nxt = (pos[0] + dr, pos[1] + dc)
        return nxt if self.in_bounds(nxt) else None

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def _cost_to_go(self) -> dict[Pos, float]:
        # Dijkstra from the goal: minimal accumulated cost (negative return) from each cell to the goal
        cost, heap = {self.goal: 0.0}, [(0.0, self.goal)]
        while heap:
            d, pos = heapq.heappop(heap)
            if d > cost[pos]:
                continue
            for dr, dc in MOVES.values():
                prev = (pos[0] - dr, pos[1] - dc)
                new = d - self.step_reward(pos)  # cost of stepping from prev into pos
                if self.in_bounds(prev) and new < cost.get(prev, float('inf')):
                    cost[prev] = new
                    heapq.heappush(heap, (new, prev))
        return cost

    def render(self, path: Optional[set[Pos]] = None) -> str:
        # ASCII view: R robot, S start, G goal, ~ rug, * path
        path = path or set()

        def cell(p: Pos) -> str:
            if p == self.pos:
                return 'R'
            if p in (self.start, self.goal):
                return 'S' if p == self.start else 'G'
            if p in path:
                return '*'
            return '~' if p in self.rug else '.'

        return '\n'.join(' '.join(cell((r, c)) for c in range(self.size)) for r in range(self.size))


class Oracle:
    """Informed policy: reads the hidden `hurry` variable. Only data generation and the human may use it."""

    def __init__(self, env: LivingRoom, rng: Optional[random.Random] = None):
        self.env = env
        self.rng = rng  # if given, ties between optimal actions are broken at random

    def optimal_actions(self, pos: Optional[Pos] = None) -> list[Action]:
        # every action that starts some reward-maximising path from `pos`
        env, pos = self.env, pos or self.env.pos

        def q(a: Action) -> float:
            nxt = env.move(pos, a)
            return env._cost[nxt] - env.step_reward(nxt) if nxt is not None else float('inf')

        qs = {a: q(a) for a in Action}
        best = min(qs.values())
        return [a for a in Action if qs[a] <= best + 1e-9]

    def act(self, pos: Optional[Pos] = None) -> Action:
        options = self.optimal_actions(pos)
        return self.rng.choice(options) if self.rng else options[0]


@dataclass(frozen=True)
class Trajectory:
    states: tuple[Pos, ...]  # s_0 .. s_T
    actions: tuple[Action, ...]  # a_0 .. a_{T-1}
    ret: float
    rewards: tuple[float, ...] = ()  # r_1 .. r_T, as the agent received them


def rollout(env: LivingRoom, policy: Callable[[Pos], Action], hurry: Optional[bool] = None) -> Trajectory:
    pos, states, actions, rewards, done = env.reset(hurry), [env.start], [], [], False
    while not done:
        action = policy(pos)
        pos, reward, done, _ = env.step(action)
        states.append(pos)
        actions.append(Action(action))
        rewards.append(reward)
    return Trajectory(tuple(states), tuple(actions), env.accrued, tuple(rewards))


if __name__ == '__main__':
    env = LivingRoom()
    oracle = Oracle(env)
    for hurry in (True, False):
        traj = rollout(env, oracle.act, hurry=hurry)
        print(f'hurry={hurry}  steps={env.steps}  return={traj.ret}\n{env.render(set(traj.states[1:]))}\n')
