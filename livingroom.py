"""LivingRoom: a gridworld where a robot walks to a goal, avoiding a rug unless it is in a hurry."""
from __future__ import annotations

import heapq
import random
from enum import IntEnum
from typing import Optional

Pos = tuple[int, int]  # (row, col)


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MOVES = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}


class LivingRoom:
    def __init__(
        self,
        size: int = 20,
        rug_shape: tuple[int, int] = (12, 10),
        rug_penalty: float = 5.0,
        hurry_penalty: float = 2.0,
        max_steps: int = 200,
        max_shove: int = 3,
    ):
        self.size = size

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

        self.rug_penalty = rug_penalty
        self.hurry_penalty = hurry_penalty
        self.max_steps = max_steps
        self.max_shove = max_shove
        self.reset()

    def reset(self, hurry: Optional[bool] = None) -> Pos:
        self.hurry = random.random() < 0.2 if hurry is None else hurry
        self.pos = self.start
        self.steps = 0
        self._cost = self._cost_to_go()
        return self.pos

    def step(self, action: Action) -> tuple[Pos, float, bool, dict]:
        dr, dc = MOVES[Action(action)]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        if self.in_bounds(nxt):
            self.pos = nxt

        self.steps += 1
        info = {'hurry': self.hurry}
        return self.pos, self.reward(self.pos), self.done, info

    def shove(self, direction: Action, distance: int) -> Pos:
        # external push by a human: moves the robot up to `max_shove` cells (stopping at walls),
        # without counting as a robot step or yielding reward
        if not 1 <= distance <= self.max_shove:
            raise ValueError(f'shove distance must be in [1, {self.max_shove}], got {distance}')
        dr, dc = MOVES[Action(direction)]
        for _ in range(distance):
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            if not self.in_bounds(nxt):
                break
            self.pos = nxt
        return self.pos

    @property
    def done(self) -> bool:
        return self.pos == self.goal or self.steps >= self.max_steps

    def reward(self, pos: Pos) -> float:
        # -1 per step; in a hurry every step costs extra, otherwise standing on the rug does.
        if self.hurry:
            return -1.0 - self.hurry_penalty
        return -1.0 - (self.rug_penalty if pos in self.rug else 0.0)

    def optimal_action(self, pos: Optional[Pos] = None) -> Action:
        # greedy w.r.t. step cost + cost-to-go, i.e. the reward-maximising policy
        pos = pos or self.pos

        def q(a: Action) -> float:
            nxt = (pos[0] + MOVES[a][0], pos[1] + MOVES[a][1])
            return self._cost[nxt] - self.reward(nxt) if self.in_bounds(nxt) else float('inf')

        return min(Action, key=q)

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
                new = d - self.reward(pos)  # cost of stepping from prev into pos
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


if __name__ == '__main__':
    env = LivingRoom()
    for hurry in (True, False):
        pos, path, total, done = env.reset(hurry=hurry), set(), 0.0, False
        while not done:
            pos, reward, done, _ = env.step(env.optimal_action())
            path.add(pos)
            total += reward
        print(f'hurry={hurry}  steps={env.steps}  return={total}\n{env.render(path)}\n')
