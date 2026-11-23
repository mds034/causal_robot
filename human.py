"""Humans who see the hidden `hurry` variable and correct the robot with a single shove.

Shoves are 3 cells by default. `ShoveDistance` draws random shove lengths instead, from a normal distribution
or uniformly over 1 to `max_shove` cells.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Optional

from livingroom import Action, LivingRoom, Oracle, Pos


class ScriptedHuman:
    """Shoves the robot `distance` cells in `direction` when it is at `origin`, at most once per episode.

    The human is rational: it only shoves when every forced step is optimal under the true `hurry`. With the
    defaults (3 cells right from the dock, onto the rug) that means it shoves exactly when the robot is in a
    hurry. Varying the shove's place, direction and timing is future work.
    """

    def __init__(self, env: LivingRoom, origin: Optional[Pos] = None, direction: Action = Action.RIGHT,
                 distance: int = 3):
        self.env = env
        self.oracle = Oracle(env)
        self.origin = origin or env.start
        self.direction, self.distance = direction, distance
        self.used = False

    def reset(self) -> None:
        self.used = False

    def intervene(self, intended: Optional[Action] = None) -> Optional[tuple[Action, int]]:
        env = self.env
        if self.used or env.done or env.pos != self.origin or not self.shove_is_optimal():
            return None
        self.used = True
        return self.direction, self.distance

    def shove_is_optimal(self) -> bool:
        pos = self.origin
        for _ in range(self.distance):
            if pos == self.env.goal or self.direction not in self.oracle.optimal_actions(pos):
                return False
            pos = self.env.move(pos, self.direction)
        return True


class ShoveDistance:
    """Random shove lengths in whole cells.

    `kind='normal'` draws from Normal(mean, std) and rounds. `kind='uniform'` draws every length from `low` to
    `high` (1 to the room's `max_shove`) with equal probability. `sample(check)` redraws until `check(distance)` returns None (acceptable), and counts why draws were
    rejected (the reasons `check` returns). Every raw draw is kept in `draws` so the distribution can be checked.
    """

    def __init__(self, mean: float = 3.0, std: float = 1.0, rng: Optional[random.Random] = None,
                 max_tries: int = 1000, kind: str = 'normal', low: int = 1, high: int = 6):
        if kind not in ('normal', 'uniform'):
            raise ValueError(f'unknown shove length distribution {kind!r}')
        self.mean, self.std, self.max_tries = mean, std, max_tries
        self.kind, self.low, self.high = kind, low, high
        self.rng = rng or random.Random()
        self.draws: list[int] = []
        self.rejected: Counter[str] = Counter()

    def draw(self) -> int:
        if self.kind == 'uniform':
            d = self.rng.randint(self.low, self.high)
        else:
            d = round(self.rng.gauss(self.mean, self.std))
        self.draws.append(d)
        return d

    def expected(self) -> tuple[float, float]:
        # the mean and std the raw draws should have
        if self.kind == 'uniform':
            n = self.high - self.low + 1
            return (self.low + self.high) / 2, ((n * n - 1) / 12) ** 0.5
        return self.mean, self.std

    def describe(self) -> str:
        return f'Uniform({self.low}..{self.high})' if self.kind == 'uniform' else f'Normal({self.mean}, {self.std})'

    def sample(self, check: Callable[[int], Optional[str]]) -> Optional[int]:
        for _ in range(self.max_tries):
            d = self.draw()
            problem = check(d)
            if problem is None:
                return d
            self.rejected[problem] += 1
        return None  # nothing acceptable here (e.g. the robot is against a wall in that direction)


def shove_check(env: LivingRoom, oracle: Oracle, pos: Pos, direction: Action) -> Callable[[int], Optional[str]]:
    # a shove of a given length from `pos` must fit in the room and, for a rational human, be optimal throughout
    def check(distance: int) -> Optional[str]:
        problem = env.shove_problem(pos, direction, distance)
        if problem is None and not shove_is_optimal(env, oracle, pos, direction, distance):
            problem = 'not_optimal'
        return problem
    return check


class RandomShoveHuman:
    """Shoves the robot `distance` cells once, at a chosen or random moment when it is about to go wrong.

    The human is rational: it only steps in when the robot's intended action is suboptimal under the true
    `hurry`, and only shoves in a direction where all forced steps are optimal. Such moments are "shove
    opportunities", found using shoves of `opportunity_distance` cells (3) whatever `distance` is, so `at`
    names the same moment for every shove length. `at` picks which opportunity (0 = the first). `direction`
    picks the direction; if it is not given or not valid there, one of the valid directions is drawn from `rng`.
    """

    def __init__(self, env: LivingRoom, at: int = 0, direction: Optional[Action] = None, distance: int = 3,
                 rng: Optional[random.Random] = None, opportunity_distance: int = 3):
        self.env = env
        self.oracle = Oracle(env)
        self.at, self.direction, self.distance = at, direction, distance
        self.opportunity_distance = opportunity_distance
        self.rng = rng or random.Random()
        self.reset()

    def reset(self) -> None:
        self.used, self.seen = False, 0

    def valid_directions(self, pos=None) -> list[Action]:
        pos = pos or self.env.pos
        return [d for d in Action if shove_is_optimal(self.env, self.oracle, pos, d, self.opportunity_distance)]

    def is_opportunity(self, intended: Action) -> bool:
        env = self.env
        return (not env.done and intended not in self.oracle.optimal_actions()
                and bool(self.valid_directions()))

    def intervene(self, intended: Optional[Action] = None) -> Optional[tuple[Action, int]]:
        if self.used or intended is None or not self.is_opportunity(intended):
            return None
        k, self.seen = self.seen, self.seen + 1
        if k != self.at:
            return None
        options = self.valid_directions()
        direction = self.direction if self.direction in options else self.rng.choice(options)
        if shove_check(self.env, self.oracle, self.env.pos, direction)(self.distance):
            return None  # this length does not fit here, or would not be optimal: no shove
        self.used = True
        return direction, self.distance


def shove_is_optimal(env: LivingRoom, oracle: Oracle, pos, direction: Action, distance: int) -> bool:
    for _ in range(distance):
        if pos == env.goal or direction not in oracle.optimal_actions(pos):
            return False
        pos = env.move(pos, direction)
    return True
