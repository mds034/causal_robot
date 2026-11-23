"""Agents that cannot see the hidden `hurry` variable.

- QLearningAgent: standard tabular RL. It optimises expected return over `hurry`, so it always follows the
  most commonly optimal route (around the rug) and has no way to use a human correction.
- MemoryAgent: remembers every optimal trajectory seen in training. After a human shove it uses those prior
  trajectories to decide what is optimal from where it landed: it keeps only the remembered trajectories that
  contain the shove and follows one of them. If none contain it, it follows the group of trajectories the
  shove pushed it towards.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from typing import Iterable, Optional

from livingroom import MOVES, Action, LivingRoom, Pos, Trajectory


class QLearningAgent:
    name = 'q-learning'

    def __init__(self, env: LivingRoom, alpha: float = 0.1, epsilon: float = 0.1, seed: int = 0):
        self.env = env
        self.alpha, self.epsilon = alpha, epsilon
        self.rng = random.Random(seed)
        self.q: dict[tuple[Pos, Action], float] = {}  # optimistic 0 init: every true value is negative

    def train(self, episodes: int) -> None:
        env = self.env
        for _ in range(episodes):
            pos, done = env.reset(), False  # `hurry` is sampled by the env and never shown to the agent
            while not done:
                a = self.rng.choice(list(Action)) if self.rng.random() < self.epsilon else self.act(pos)
                nxt, reward, done, _ = env.step(a)
                target = reward + (0.0 if nxt == env.goal else max(self.value(nxt, b) for b in Action))
                self.q[pos, a] = self.value(pos, a) + self.alpha * (target - self.value(pos, a))
                pos = nxt

    def value(self, pos: Pos, a: Action) -> float:
        return self.q.get((pos, a), 0.0)

    def reset(self) -> None:
        pass

    def act(self, pos: Pos) -> Action:
        return max(Action, key=lambda a: self.value(pos, a))

    def observe_correction(self, origin: Pos, direction: Action, distance: int, dest: Pos) -> None:
        pass  # a standard RL policy has nothing to update from a single deployment-time correction

    def candidate_routes(self) -> list[Trajectory]:
        return []


class MemoryAgent:
    """Keeps every optimal trajectory seen in training, grouped into presumed latent settings.

    The agent never sees `hurry`. It groups trajectories by return instead: under one setting of the latent
    variables every optimal trajectory has the same return, so (in this environment) equal returns identify a
    latent setting. Groups are ordered by how often they were seen in training.
    """

    name = 'memory'

    def __init__(self, env: LivingRoom):
        self.env = env
        self.groups: dict[float, Counter[Trajectory]] = {}
        self.policy: dict[Pos, Action] = {}
        self.active: list[Trajectory] = []
        self.last_choice = ''  # how the last correction was handled: 'filter' or 'fallback'

    def train(self, demos: Iterable[Trajectory]) -> None:
        for traj in demos:
            self.groups.setdefault(traj.ret, Counter())[traj] += 1
        self.reset()

    def group_count(self, ret: float) -> int:
        return sum(self.groups[ret].values())

    def reset(self) -> None:
        # with no correction, behave like standard RL: follow the most common route of the most common group
        ret = max(self.groups, key=self.group_count)
        self.follow(ret, list(self.groups[ret]))

    def follow(self, ret: float, routes: list[Trajectory], near: Optional[Pos] = None) -> None:
        # follow the most common route, or with `near` the route closest to that cell (then the most common)
        self.active = sorted(routes, key=lambda t: (distance_to(near, t.states) if near else 0, -self.groups[ret][t]))
        best = self.active[0]
        self.policy = dict(zip(best.states, best.actions))

    def act(self, pos: Pos) -> Action:
        if pos in self.policy:
            return self.policy[pos]
        return self.rejoin(pos)

    def rejoin(self, pos: Pos) -> Action:
        # off the remembered route: take the first step of a shortest path back onto it
        first: dict[Pos, Optional[Action]] = {pos: None}
        queue = deque([pos])
        while queue:
            cur = queue.popleft()
            if cur in self.policy:
                return first[cur]
            for a in Action:
                nxt = self.env.move(cur, a)
                if nxt is not None and nxt not in first:
                    first[nxt] = first[cur] if first[cur] is not None else a
                    queue.append(nxt)
        raise RuntimeError('no remembered route is reachable')

    def observe_correction(self, origin: Pos, direction: Action, distance: int, dest: Pos) -> None:
        # Keep only the remembered optimal trajectories that contain every cell of the shove.
        segment = segment_states(origin, direction, dest)
        consistent = {ret: [t for t in routes if contains(t, segment)] for ret, routes in self.groups.items()}
        consistent = {ret: ts for ret, ts in consistent.items() if ts}
        if consistent:
            ret = max(consistent, key=self.group_count)
            self.follow(ret, consistent[ret])
            self.last_choice = 'filter'
            return

        # No remembered optimal trajectory contains the shove. The human is rational and sees the latent
        # variables, so the shove's direction points towards the right routes: score each group by how much
        # closer the shove moved the robot to its remembered routes. Among the best-scoring (valid) groups,
        # take the highest return, then rejoin that group's route nearest to `dest`.
        progress = {ret: shove_progress(segment, routes) for ret, routes in self.groups.items()}
        valid = [ret for ret, p in progress.items() if p == max(progress.values())]
        ret = max(valid)
        self.follow(ret, list(self.groups[ret]), near=dest)
        self.last_choice = 'toward'

    def candidate_routes(self) -> list[Trajectory]:
        return self.active


def distance_to(pos: Pos, cells: Iterable[Pos]) -> int:
    return min(abs(pos[0] - r) + abs(pos[1] - c) for r, c in cells)


def shove_progress(segment: list[Pos], routes: Iterable[Trajectory]) -> int:
    # how many cells closer to the routes the shove moved the robot (negative: it moved away)
    cells = {s for t in routes for s in t.states}
    return distance_to(segment[0], cells) - distance_to(segment[-1], cells)


def segment_states(origin: Pos, direction: Action, dest: Pos) -> list[Pos]:
    dr, dc = MOVES[direction]
    states = [origin]
    while states[-1] != dest:
        states.append((states[-1][0] + dr, states[-1][1] + dc))
    return states


def contains(traj: Trajectory, segment: list[Pos]) -> bool:
    n = len(segment)
    return any(list(traj.states[i:i + n]) == segment for i, s in enumerate(traj.states) if s == segment[0])
