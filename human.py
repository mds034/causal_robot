"""The human: sees the hidden `hurry` variable and shoves the robot at most once per episode.

How the shove depends on `hurry`:

- In a hurry: early in the episode (step drawn from Normal(2, 1), clipped to the first 5 steps), always to
  the right, a uniform 1..`max_shove` cells. If the rug lies to the robot's right in its row, the shove is
  lengthened to reach it.
- Not in a hurry: only while the robot is on the rug. A delay drawn from Normal(3, 1.5), clipped to 0..6
  steps, starts the first time the robot steps on the rug. When it has passed and the robot is on the rug,
  the human pushes it off by the shortest way: towards the nearest rug edge (ties broken at random), just far
  enough to leave the rug (or `max_shove` cells if that is not enough).
"""
from __future__ import annotations

import random
from typing import Optional

from livingroom import MOVES, Action, LivingRoom


class Human:
    HURRY_MEAN, HURRY_STD, HURRY_LAST = 2.0, 1.0, 4  # shove step, clipped to 0..HURRY_LAST
    CALM_MEAN, CALM_STD, CALM_LAST = 3.0, 1.5, 6  # delay after first stepping on the rug, clipped to 0..CALM_LAST

    def __init__(self, env: LivingRoom, rng: Optional[random.Random] = None):
        self.env = env
        self.rng = rng or random.Random()
        self.used, self.first_on_rug, self.due = False, None, None  # `reset()` draws the timing

    @classmethod
    def settings(cls) -> dict:
        return {
            'hurry': {'step': f'clip(round(Normal({cls.HURRY_MEAN}, {cls.HURRY_STD})), 0, {cls.HURRY_LAST})',
                      'direction': 'RIGHT', 'distance': 'Uniform(1..max_shove), lengthened to reach the rug'},
            'not_hurry': {'delay_after_rug': f'clip(round(Normal({cls.CALM_MEAN}, {cls.CALM_STD})), 0, {cls.CALM_LAST})',
                          'direction': 'nearest rug edge (random tie-break)',
                          'distance': 'just off the rug, or max_shove'},
        }

    def reset(self) -> None:
        # call after env.reset(), once per episode: the timing drawn depends on the episode's `hurry`
        self.used, self.first_on_rug = False, None
        if self.env.hurry:
            self.due = self.clipped(self.HURRY_MEAN, self.HURRY_STD, self.HURRY_LAST)
        else:
            self.due = self.clipped(self.CALM_MEAN, self.CALM_STD, self.CALM_LAST)

    def clipped(self, mean: float, std: float, last: int) -> int:
        return min(last, max(0, round(self.rng.gauss(mean, std))))

    def intervene(self, intended: Optional[Action] = None) -> Optional[tuple[Action, int]]:
        # called before each of the robot's actions; `intended` is accepted for the callers but not used
        env = self.env
        if self.due is None:
            self.reset()  # not reset since this episode began
        if self.used or env.done:
            return None
        shove = self.hurry_shove() if env.hurry else self.calm_shove()
        if shove:
            self.used = True
        return shove

    def hurry_shove(self) -> Optional[tuple[Action, int]]:
        env = self.env
        if env.steps < self.due:
            return None
        distance = self.rng.randint(1, env.max_shove)
        gap = self.gap_to_rug(Action.RIGHT)
        if gap is not None:
            distance = max(distance, gap)
        r, c = env.pos
        distance = min(distance, env.max_shove, env.size - 1 - c)
        return (Action.RIGHT, distance) if distance >= 1 else None

    def calm_shove(self) -> Optional[tuple[Action, int]]:
        env = self.env
        if env.pos not in env.rug:
            return None
        if self.first_on_rug is None:
            self.first_on_rug = env.steps
        if env.steps - self.first_on_rug < self.due:
            return None
        exits = {a: n for a in Action if (n := self.cells_to_leave_rug(a)) is not None}
        nearest = min(exits.values())
        direction = self.rng.choice([a for a in Action if exits.get(a) == nearest])
        return direction, min(nearest, env.max_shove)

    def gap_to_rug(self, direction: Action) -> Optional[int]:
        # cells to the first rug cell in `direction` (from off the rug), or None if there is none
        env, (r, c) = self.env, self.env.pos
        if (r, c) in env.rug:
            return None
        dr, dc = MOVES[direction]
        n = 1
        while env.in_bounds((r + dr * n, c + dc * n)):
            if (r + dr * n, c + dc * n) in env.rug:
                return n
            n += 1
        return None

    def cells_to_leave_rug(self, direction: Action) -> Optional[int]:
        # cells to the first off-rug cell in `direction` (from on the rug), or None if the room ends first
        env, (r, c) = self.env, self.env.pos
        dr, dc = MOVES[direction]
        n = 1
        while env.in_bounds((r + dr * n, c + dc * n)):
            if (r + dr * n, c + dc * n) not in env.rug:
                return n
            n += 1
        return None
