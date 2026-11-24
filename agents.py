"""Agents that cannot see the hidden `hurry` variable. Both learn with the same tabular Q-learning.

Both agents run online Q-learning in training, keep every transition they experienced (with how often they saw
it), and after training replay that experience through the same Q-learning update until the values settle.

- QLearningAgent: the standard RL baseline. It replays all its experience together, so it optimises expected
  return over everything it experienced, and has no way to use a human correction.
- MemoryAgent: the same, plus its experience grouped by the rewards each episode paid. Episodes that paid
  different rewards for the same cell go into different groups; the agent is never told why, or how many
  groups to expect. It replays each group separately, giving one Q-function per group. Without a correction it
  acts exactly like the baseline. After a human shove it follows the group whose greedy moves agree with it.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Optional

from livingroom import MOVES, Action, LivingRoom, Obs, Pos, Trajectory

Transition = tuple[Pos, Action, float, Pos]  # (s, a, r, s')
Values = dict[tuple[Pos, Action], float]


def replay_values(experience: Counter[Transition], alpha: float, goal: Pos, tol: float = 1e-6,
                  max_sweeps: int = 20000) -> Values:
    """Replay experience through the Q-learning update until the values settle.

    Each (s, a, s') is replayed with its rewards averaged by how often each was seen, so the values are those
    of the experience as it was sampled. Only the (s, a) pairs in `experience` get values; a move into a cell
    with no experienced actions leads nowhere known and is marked unusable (-inf).
    """
    seen_r: dict[tuple[Pos, Action, Pos], list[float]] = defaultdict(lambda: [0.0, 0])
    for (s, a, r, nxt), n in experience.items():
        acc = seen_r[s, a, nxt]
        acc[0] += r * n
        acc[1] += n
    moves = sorted((s, a, total / n, nxt) for (s, a, nxt), (total, n) in seen_r.items())  # fixed order
    q: Values = {(s, a): 0.0 for s, a, _, _ in moves}
    actions: dict[Pos, list[Action]] = {}
    for s, a in q:
        actions.setdefault(s, []).append(a)
    dead = float('-inf')
    for _ in range(max_sweeps):
        change = 0.0
        for s, a, r, nxt in moves:
            if q[s, a] == dead:
                continue
            future = 0.0 if nxt == goal else max((q[nxt, b] for b in actions.get(nxt, ())), default=dead)
            if future == dead:
                q[s, a], change = dead, max(change, 1.0)
                continue
            delta = alpha * (r + future - q[s, a])
            q[s, a] += delta
            change = max(change, abs(delta))
        if change < tol:
            break
    return q


def greedy_actions(values: Values, pos: Pos) -> list[Action]:
    # every best action at `pos` under `values` (empty if there is no experience of acting there)
    seen = [a for a in Action if (pos, a) in values]
    if not seen:
        return []
    best = max(values[pos, a] for a in seen)
    return [a for a in seen if values[pos, a] >= best - 1e-9]


class QLearningAgent:
    name = 'q-learning'

    def __init__(self, env: LivingRoom, alpha: float = 0.1, epsilon: float = 0.1, seed: int = 0):
        self.env = env
        self.alpha, self.epsilon = alpha, epsilon
        self.rng = random.Random(seed)
        self.q: dict[tuple[Obs, Action], float] = {}  # online values; optimistic 0 init: every true value is negative
        self.experience: Counter[Transition] = Counter()  # every transition seen in training, with its count
        self.values: Values = {}  # from replaying `experience` after training; used to act once set

    def train(self, episodes: int) -> None:
        env = self.env
        for _ in range(episodes):
            env.reset()  # `hurry` is sampled by the env and never shown to the agent
            obs, done = env.observe(), False
            states, actions, rewards = [obs.pos], [], []
            while not done:
                a = self.rng.choice(list(Action)) if self.rng.random() < self.epsilon else self.act(obs)
                _, reward, done, _ = env.step(a)
                nxt = env.observe()
                target = reward + (0.0 if nxt.pos == nxt.goal else max(self.value(nxt, b) for b in Action))
                self.q[obs, a] = self.value(obs, a) + self.alpha * (target - self.value(obs, a))
                obs = nxt
                states.append(obs.pos)
                actions.append(Action(a))
                rewards.append(reward)
            self.remember(Trajectory(tuple(states), tuple(actions), env.accrued, tuple(rewards)))
        self.replay()

    def remember(self, traj: Trajectory) -> None:
        self.experience.update(zip(traj.states, traj.actions, traj.rewards, traj.states[1:]))

    def replay(self) -> None:
        self.values = replay_values(self.experience, self.alpha, self.env.observe().goal)

    def value(self, obs: Obs, a: Action) -> float:
        return self.q.get((obs, a), 0.0)

    def reset(self) -> None:
        pass

    def act(self, obs: Obs) -> Action:
        # during training (no replayed values yet) the online Q-table; afterwards the replayed values, which in
        # this room are keyed by position since position determines the rest of the observation
        best = greedy_actions(self.values, obs.pos)
        if best:
            return best[0]
        return max(Action, key=lambda a: self.value(obs, a))

    def observe_correction(self, origin: Pos, direction: Action, distance: int, dest: Pos) -> None:
        pass  # a standard RL policy has nothing to update from a single deployment-time correction

    def candidate_routes(self) -> list[Trajectory]:
        return []


class Group:
    """Training episodes that never paid different rewards for the same cell, and their experience."""

    def __init__(self):
        self.rewards: dict[Pos, float] = {}  # reward paid for entering each cell, in every episode of the group
        self.transitions: Counter[Transition] = Counter()
        self.episodes = 0
        self.q: Values = {}  # from replaying `transitions`; defined only where the group has experience

    def agreement(self, rewards: dict[Pos, float]) -> Optional[int]:
        # cells on which this group and an episode agree, or None if they disagree on any cell
        shared = rewards.keys() & self.rewards.keys()
        if any(rewards[c] != self.rewards[c] for c in shared):
            return None
        return len(shared)

    def add(self, rewards: dict[Pos, float], transitions: list[Transition]) -> None:
        self.rewards.update(rewards)
        self.transitions.update(transitions)
        self.episodes += 1

    def greedy(self, pos: Pos) -> list[Action]:
        return greedy_actions(self.q, pos)


class MemoryAgent(QLearningAgent):
    """The baseline, plus its experience grouped by the rewards episodes paid.

    `train` is the baseline's (online Q-learning, then replaying all experience), so without a correction this
    agent is the baseline. Each finished training episode also joins the group it agrees with on the most cells
    (and never disagrees with), or starts a new group. After training, each group's experience is replayed
    separately with the same function, giving one Q-function per group. A shove is a sequence of
    (cell, direction) pairs; the agent follows the group for which every one of them is a greedy action (the
    most common such group if several are).
    """

    name = 'memory'

    def __init__(self, env: LivingRoom, alpha: float = 0.1, epsilon: float = 0.1, seed: int = 0):
        super().__init__(env, alpha, epsilon, seed)
        self.groups: list[Group] = []
        self.episode_groups: list[int] = []  # the group each training episode joined, in order
        self.chosen: Optional[int] = None  # the group followed after a correction
        self.last_choice = ''  # how the last correction was handled: 'filter' or 'no-match'

    def train(self, episodes: int) -> None:
        super().train(episodes)
        self.learn_groups()

    def remember(self, traj: Trajectory) -> None:
        super().remember(traj)
        rewards = dict(zip(traj.states[1:], traj.rewards))  # reward paid for entering each cell
        transitions = list(zip(traj.states, traj.actions, traj.rewards, traj.states[1:]))
        scores = [g.agreement(rewards) for g in self.groups]
        fits = [i for i, s in enumerate(scores) if s]  # agrees on at least one cell and disagrees on none
        if fits:
            k = max(fits, key=lambda i: scores[i])
        else:
            self.groups.append(Group())
            k = len(self.groups) - 1
        self.groups[k].add(rewards, transitions)
        self.episode_groups.append(k)
        # groups that never overlapped can turn out to be one: this episode agrees with both, and they agree
        # with each other wherever they overlap
        for i in sorted((i for i in fits if i != k), reverse=True):
            if self.groups[k].agreement(self.groups[i].rewards) is not None:
                k = self.merge(i, into=k)

    def merge(self, i: int, into: int) -> int:
        # fold group `i` into group `into`; returns the new index of `into`
        src, dst = self.groups[i], self.groups[into]
        dst.rewards.update(src.rewards)
        dst.transitions.update(src.transitions)
        dst.episodes += src.episodes
        del self.groups[i]
        self.episode_groups = [into if g == i else g for g in self.episode_groups]
        self.episode_groups = [g - 1 if g > i else g for g in self.episode_groups]
        return into - 1 if into > i else into

    def learn_groups(self) -> None:
        goal = self.env.observe().goal
        for g in self.groups:
            g.q = replay_values(g.transitions, self.alpha, goal)

    def reset(self) -> None:
        self.chosen = None

    def act(self, obs: Obs) -> Action:
        if self.chosen is not None:
            best = self.groups[self.chosen].greedy(obs.pos)
            if best:
                return best[0]
        return super().act(obs)

    def consistent_groups(self, origin: Pos, direction: Action, dest: Pos) -> list[int]:
        # groups for which every (cell, direction) pair of the shove is a greedy action
        pushed = segment_states(origin, direction, dest)[:-1]
        return [i for i, g in enumerate(self.groups) if all(direction in g.greedy(cell) for cell in pushed)]

    def observe_correction(self, origin: Pos, direction: Action, distance: int, dest: Pos) -> None:
        matches = self.consistent_groups(origin, direction, dest)
        if not matches:
            self.chosen = None  # no past experience agrees with the shove: act like the baseline
            self.last_choice = 'no-match'
            return
        self.chosen = max(matches, key=lambda i: self.groups[i].episodes)
        self.last_choice = 'filter'

    def candidate_routes(self) -> list[Trajectory]:
        # the followed group's greedy path from the robot's position, for drawing
        if self.chosen is None:
            return []
        g, pos, goal = self.groups[self.chosen], self.env.pos, self.env.observe().goal
        states, actions = [pos], []
        while pos != goal and len(states) <= self.env.max_steps:
            best = g.greedy(pos)
            nxt = self.env.move(pos, best[0]) if best else None
            if nxt is None:
                break
            states.append(nxt)
            actions.append(best[0])
            pos = nxt
        return [Trajectory(tuple(states), tuple(actions), 0.0)]


def segment_states(origin: Pos, direction: Action, dest: Pos) -> list[Pos]:
    dr, dc = MOVES[direction]
    states = [origin]
    while states[-1] != dest:
        states.append((states[-1][0] + dr, states[-1][1] + dc))
    return states
