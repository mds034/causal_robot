"""Train both agents without access to `hurry`, then deploy them with a human who may shove once (3 cells).

Every run gets an ID (m1, m2, ... for the memory agent, q1, q2, ... for Q-learning) that `gui.py --replay`
accepts, so any row printed here can be watched in the GUI.

Results are saved to results/seed<S>_demos<D>_q<Q>_shove-<lengths>-draws<N>/ (<lengths> is uniform1-6, or
mean<M>-std<SD> for normal lengths): runs.csv has one row per run, settings.json the settings, environment, memory groups and git commit they came from, and shove_distance_check.json the validation of the
random shove lengths. Rerunning with the same settings overwrites them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents import MemoryAgent, QLearningAgent
from human import RandomShoveHuman, ScriptedHuman, ShoveDistance, shove_check
from livingroom import MOVES, Action, LivingRoom, Oracle, rollout

DEMOS, Q_EPISODES, SEED = 200, 30000, 0
SHOVE_DIST, SHOVE_MEAN, SHOVE_STD, DISTANCE_DRAWS = 'uniform', 3.0, 1.0, 5  # random shove lengths; draws per placement
CACHE_DIR = Path(__file__).parent / '.cache'
RESULTS_DIR = Path(__file__).parent / 'results'
COLUMNS = ('id', 'agent', 'hurry', 'shove', 'at', 'direction', 'distance', 'origin', 'dest', 'handled', 'return',
           'optimal', 'regret', 'best_after_shove', 'regret_after_shove', 'reached_goal')
PREFIX = {'memory': 'm', 'q-learning': 'q'}


def train_memory(env: LivingRoom, demos: int = DEMOS, seed: int = SEED) -> MemoryAgent:
    random.seed(seed)  # the env samples `hurry` from the global RNG
    oracle = Oracle(env, rng=random.Random(seed))  # random tie-breaking shows many distinct optimal routes
    memory = MemoryAgent(env)
    memory.train(rollout(env, oracle.act) for _ in range(demos))
    return memory


def train_q(env: LivingRoom, episodes: int = Q_EPISODES, seed: int = SEED, cache: bool = True) -> QLearningAgent:
    # Q-learning takes ~25s, so the learned table is cached per (env, episodes, seed)
    config = repr((env.size, sorted(env.rug), env.rug_penalty, env.hurry_penalty, env.max_steps, episodes, seed))
    path = CACHE_DIR / f'q-{hashlib.sha1(config.encode()).hexdigest()[:12]}.pkl'
    q = QLearningAgent(env, seed=seed)
    if cache and path.exists():
        q.q = pickle.loads(path.read_bytes())
        return q
    random.seed(seed)
    q.train(episodes)
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(q.q))
    return q


def train(name: str, env: LivingRoom, demos: int = DEMOS, q_episodes: int = Q_EPISODES, seed: int = SEED):
    return train_memory(env, demos, seed) if name == 'memory' else train_q(env, q_episodes, seed)


@dataclass(frozen=True)
class Run:
    """One deployment episode: which agent, the true `hurry`, and how (if at all) the human shoves."""

    id: str
    agent: str  # 'memory' or 'q-learning'
    hurry: bool
    # 'none', 'fixed' (3 right from the dock), 'random' (3 cells at opportunity `at`, in `direction`) or
    # 'random-distance' (the same placements, `distance` cells drawn from a normal distribution)
    shove: str
    at: Optional[int] = None
    direction: Optional[Action] = None
    distance: Optional[int] = None

    def human(self, env: LivingRoom):
        if self.shove == 'fixed':
            return ScriptedHuman(env)
        if self.shove in ('random', 'random-distance'):
            return RandomShoveHuman(env, at=self.at, direction=self.direction, distance=self.distance)
        return None

    def describe(self) -> str:
        hurry = 'in a hurry' if self.hurry else 'not in a hurry'
        direction = self.direction.name if self.direction is not None else 'random direction'
        shove = {'none': 'no shove', 'fixed': 'fixed shove (3 right from the dock)'}.get(
            self.shove, f'shove at opportunity {self.at}, {direction} {self.distance}')
        return f'{self.id}: {self.agent}, {hurry}, {shove}'


def shove_lengths(env: LivingRoom, dist: str, mean: float, std: float, rng: random.Random) -> ShoveDistance:
    # uniform lengths run from 1 to the longest shove the room allows
    return ShoveDistance(mean, std, rng=rng, kind=dist, low=1, high=env.max_shove)


def plan_runs(env: LivingRoom, agent, seed: int = SEED, mean: float = SHOVE_MEAN, std: float = SHOVE_STD,
              draws: int = DISTANCE_DRAWS, dist: str = SHOVE_DIST) -> list[Run]:
    # the fixed-shove grid, one hurried run per possible random shove placement (3 cells), then `draws` runs per
    # placement with random shove lengths (uniform, or Normal(mean, std)). Lengths are drawn here, from a seeded RNG, so
    # every run (and its ID) is the same each time the plan is made.
    prefix, runs = PREFIX[agent.name], []
    for hurry in (False, True):
        for shove in ('none', 'fixed'):
            runs.append(Run(f'{prefix}{len(runs) + 1}', agent.name, hurry, shove))
    spots = shove_opportunities(env, agent, hurry=True)
    for k, _, directions in spots:
        for d in directions:
            runs.append(Run(f'{prefix}{len(runs) + 1}', agent.name, True, 'random', k, d, 3))
    lengths = shove_lengths(env, dist, mean, std, random.Random(f'{seed}-{agent.name}-distance'))
    env.reset(hurry=True)  # the human's optimality check uses the true (hurried) costs
    oracle = Oracle(env)
    for k, pos, directions in spots:
        for d in directions:
            for _ in range(draws):
                dist = lengths.sample(shove_check(env, oracle, pos, d))
                if dist is not None:
                    runs.append(Run(f'{prefix}{len(runs) + 1}', agent.name, True, 'random-distance', k, d, dist))
    return runs


def run_episode(env: LivingRoom, agent, hurry: bool, human=None) -> dict:
    env.reset(hurry)
    agent.reset()
    if human:
        human.reset()
    total, corrections, shove, best_after = 0.0, 0, None, None
    while not env.done:
        intended = agent.act(env.pos)
        correction = human.intervene(intended) if human else None
        if correction:
            origin = env.pos
            dest, reward, _ = env.shove(*correction)
            total += reward
            corrections += 1
            shove = (origin, correction[0], dest)
            best_after = total - env._cost[dest]  # return so far plus the best possible from where it landed
            agent.observe_correction(origin, *correction, dest)
            continue
        _, reward, _, _ = env.step(intended)
        total += reward
    return {'return': total, 'optimal': env.optimal_return(), 'corrections': corrections,
            'reached_goal': env.pos == env.goal, 'shove': shove, 'best_after_shove': best_after,
            'choice': getattr(agent, 'last_choice', '-') if corrections else '-'}


def replay(env: LivingRoom, agent, run: Run) -> dict:
    return run_episode(env, agent, run.hurry, run.human(env))


def shove_opportunities(env: LivingRoom, agent, hurry: bool) -> list[tuple[int, tuple, list]]:
    # every (opportunity index, robot position, valid directions) a rational human could shove at, found by
    # replaying the agent without help: until the shove, its path is the same as with no human
    env.reset(hurry)
    agent.reset()
    probe, found = RandomShoveHuman(env), []
    while not env.done:
        intended = agent.act(env.pos)
        if probe.is_opportunity(intended):
            found.append((len(found), env.pos, probe.valid_directions()))
        env.step(intended)
    return found


def random_shove_sweep(env: LivingRoom, agent) -> list[dict]:
    # run one hurried episode per possible shove placement (opportunity x valid direction)
    rows = []
    for run in plan_runs(env, agent):
        if run.shove == 'random':
            r = replay(env, agent, run)
            r.update(id=run.id, origin=r['shove'][0], direction=run.direction.name)
            rows.append(r)
    return rows


def segment(shove: tuple) -> list:
    # cells the robot passed through during a recorded (origin, direction, dest) shove
    origin, direction, dest = shove
    dr, dc = MOVES[direction]
    cells = [origin]
    while cells[-1] != dest:
        cells.append((cells[-1][0] + dr, cells[-1][1] + dc))
    return cells


def validate_shove_distance(env: LivingRoom, mean: float = SHOVE_MEAN, std: float = SHOVE_STD, n: int = 5000,
                            seed: int = SEED, dist: str = SHOVE_DIST) -> dict:
    """Check the random shove lengths: their distribution, and that only shoves inside the room are accepted.

    1. Raw draws should have the distribution's mean and std (for Normal(mean, std), rounded, about that mean
       and std). Uniform draws must also cover every length from 1 to max_shove about equally often.
    2. From random cells and directions, lengths are sampled with the bounds check alone. Every accepted shove
       must land inside the room exactly that many cells away when executed, and the env must refuse every
       length the check rejected.
    """
    rng = random.Random(f'{seed}-validate')
    raw = shove_lengths(env, dist, mean, std, rng)
    values = [raw.draw() for _ in range(n)]
    raw_mean = sum(values) / n
    raw_std = (sum((v - raw_mean) ** 2 for v in values) / n) ** 0.5
    want_mean, want_std = raw.expected()
    counts = Counter(values)
    if dist == 'uniform':  # every length equally likely: each count within 15% of n / (number of lengths)
        lengths = range(raw.low, raw.high + 1)
        even = set(counts) == set(lengths) and all(abs(counts[d] - n / len(lengths)) < 0.15 * n / len(lengths)
                                                   for d in lengths)
    else:
        even = True

    env.reset(hurry=True)

    def place(pos):  # put the robot at `pos` without recomputing the room's costs
        env.pos, env.steps = pos, 0

    bounded = shove_lengths(env, dist, mean, std, rng)
    accepted, landed_ok, refused, not_refused, no_fit = [], 0, 0, 0, 0
    for _ in range(n):
        pos = (rng.randrange(env.size), rng.randrange(env.size))
        direction = rng.choice(list(Action))
        if env.shove_problem(pos, direction, 1) == 'out_of_room':
            no_fit += 1  # against a wall in that direction: no length fits, so do not keep redrawing
            continue

        def check(d: int) -> Optional[str]:
            nonlocal refused, not_refused
            problem = env.shove_problem(pos, direction, d)
            if problem:  # the env itself must refuse this shove
                place(pos)
                try:
                    env.shove(direction, d)
                    not_refused += 1
                except ValueError:
                    refused += 1
            return problem

        dist = bounded.sample(check)
        if dist is None:
            no_fit += 1
            continue
        accepted.append(dist)
        place(pos)
        dest, _, _ = env.shove(direction, dist)
        dr, dc = MOVES[direction]
        landed_ok += env.in_bounds(dest) and (dest == (pos[0] + dr * dist, pos[1] + dc * dist) or dest == env.goal)
    env.reset()
    return {
        'distribution': raw.describe(), 'expected_mean': round(want_mean, 3), 'expected_std': round(want_std, 3),
        'samples': n,
        'raw_draws': {'mean': round(raw_mean, 3), 'std': round(raw_std, 3), 'min': min(values), 'max': max(values),
                      'counts': {d: counts[d] for d in sorted(counts)}},
        'bounded_sampling': {
            'placements': n, 'accepted': len(accepted), 'no_length_fits': no_fit, 'max_shove': env.max_shove,
            'accepted_mean_distance': round(sum(accepted) / len(accepted), 3) if accepted else None,
            'accepted_min': min(accepted, default=None), 'accepted_max': max(accepted, default=None),
            'rejected_draws_by_reason': dict(bounded.rejected),
            'accepted_landed_in_room': landed_ok, 'rejected_refused_by_env': refused,
            'rejected_but_env_allowed': not_refused,
        },
        'passed': bool(accepted) and landed_ok == len(accepted) and not_refused == 0
                  and abs(raw_mean - want_mean) < 0.05 and abs(raw_std - want_std) < 0.1 and even,
    }


def run_all(env: LivingRoom, agents, seed: int = SEED, mean: float = SHOVE_MEAN, std: float = SHOVE_STD,
            draws: int = DISTANCE_DRAWS, dist: str = SHOVE_DIST) -> list[dict]:
    # replay every planned run once and flatten it into a row with the COLUMNS above
    rows = []
    for agent in agents:
        for run in plan_runs(env, agent, seed, mean, std, draws, dist):
            r = replay(env, agent, run)
            origin, _, dest = r['shove'] or (None, None, None)
            best = r['best_after_shove']
            rows.append({
                'id': run.id, 'agent': run.agent, 'hurry': run.hurry, 'shove': run.shove, 'at': run.at,
                'direction': run.direction.name if run.direction is not None else None,
                'distance': len(segment(r['shove'])) - 1 if r['shove'] else None,
                'origin': origin, 'dest': dest, 'handled': r['choice'], 'return': r['return'],
                'optimal': r['optimal'], 'regret': r['optimal'] - r['return'], 'best_after_shove': best,
                'regret_after_shove': None if best is None else best - r['return'], 'reached_goal': r['reached_goal'],
            })
    return rows


def results_folder(root: Path, seed: int, demos: int, q_episodes: int, mean: float, std: float, draws: int,
                   dist: str = SHOVE_DIST, max_shove: int = 6) -> Path:
    lengths = f'uniform1-{max_shove}' if dist == 'uniform' else f'mean{mean}-std{std}'
    return root / f'seed{seed}_demos{demos}_q{q_episodes}_shove-{lengths}-draws{draws}'


def save_results(rows: list[dict], env: LivingRoom, memory: MemoryAgent, demos: int, q_episodes: int, seed: int,
                 root: Path = RESULTS_DIR, mean: float = SHOVE_MEAN, std: float = SHOVE_STD,
                 draws: int = DISTANCE_DRAWS, validation: Optional[dict] = None, dist: str = SHOVE_DIST) -> Path:
    out = results_folder(root, seed, demos, q_episodes, mean, std, draws, dist, env.max_shove)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'runs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k: '' if v is None else v for k, v in row.items()} for row in rows)
    rug_rows, rug_cols = [r for r, _ in env.rug], [c for _, c in env.rug]
    settings = {
        'seed': seed, 'demos': demos, 'q_episodes': q_episodes,
        'shove_distance': {'distribution': dist, 'draws_per_placement': draws,
                           **({'low': 1, 'high': env.max_shove} if dist == 'uniform' else {'mean': mean, 'std': std})},
        'env': {'size': env.size, 'start': env.start, 'goal': env.goal,
                'rug_rows': [min(rug_rows), max(rug_rows)], 'rug_cols': [min(rug_cols), max(rug_cols)],
                'rug_penalty': env.rug_penalty, 'hurry_penalty': env.hurry_penalty, 'max_steps': env.max_steps,
                'max_shove': env.max_shove},
        'memory_groups': {str(ret): {'demos': memory.group_count(ret), 'distinct_routes': len(ts)}
                          for ret, ts in memory.groups.items()},
        'git_commit': git_commit(),
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    (out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    if validation is not None:
        (out / 'shove_distance_check.json').write_text(json.dumps(validation, indent=2) + '\n')
    return out


def git_commit() -> Optional[str]:
    # the commit the results came from, marked dirty if there were uncommitted changes
    try:
        root = Path(__file__).parent
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '*.py'], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return commit + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demos', type=int, default=DEMOS, help='oracle demonstrations for the memory agent')
    parser.add_argument('--q-episodes', type=int, default=Q_EPISODES, help='training episodes for Q-learning')
    parser.add_argument('--seed', type=int, nargs='+', default=[SEED], help='one or more seeds to run')
    parser.add_argument('--shove-dist', choices=('uniform', 'normal'), default=SHOVE_DIST,
                        help='random shove lengths: uniform over 1..max_shove, or normal (--shove-mean, --shove-std)')
    parser.add_argument('--shove-mean', type=float, default=SHOVE_MEAN, help='mean normal shove length (cells)')
    parser.add_argument('--shove-std', type=float, default=SHOVE_STD, help='std of normal shove length (cells)')
    parser.add_argument('--distance-draws', type=int, default=DISTANCE_DRAWS,
                        help='random-length shoves per shove placement')
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR, help='where to save results')
    parser.add_argument('--no-save', action='store_true', help='print results without saving them')
    args = parser.parse_args()
    for seed in args.seed:
        run_seed(seed, args)


def run_seed(seed: int, args: argparse.Namespace) -> None:
    env = LivingRoom()
    agents = [train(name, env, args.demos, args.q_episodes, seed) for name in ('q-learning', 'memory')]
    memory = agents[1]
    mean, std, draws, dist = args.shove_mean, args.shove_std, args.distance_draws, args.shove_dist
    rows = run_all(env, agents, seed, mean, std, draws, dist)
    print(f'=== seed {seed}\nmemory groups (return: demos, distinct routes):',
          {ret: (memory.group_count(ret), len(ts)) for ret, ts in memory.groups.items()})

    print(f'\n{"id":<4} {"agent":<11} {"hurry":<6} {"shove":<6} {"return":>7} {"optimal":>8} {"regret":>7}')
    for r in rows:
        if r['shove'] in ('none', 'fixed'):
            print(f'{r["id"]:<4} {r["agent"]:<11} {r["hurry"]!s:<6} {r["shove"]:<6} {r["return"]:>7.0f} '
                  f'{r["optimal"]:>8.0f} {r["regret"]:>7.0f}')

    print('\nRandom shove placement (in a hurry): one episode per place a rational human could shove 3 cells')
    for agent in agents:
        sweep = [r for r in rows if r['agent'] == agent.name and r['shove'] == 'random']
        best = sum(r['regret_after_shove'] == 0 for r in sweep)
        mean_return = sum(r['return'] for r in sweep) / len(sweep)
        mean_best = sum(r['best_after_shove'] for r in sweep) / len(sweep)
        print(f'\n{agent.name}: {len(sweep)} placements, best possible after the shove in {best}, '
              f'mean return {mean_return:.1f} (mean best after the shove {mean_best:.1f})')
        print(f'  {"id":<4} {"from":<9} {"dir":<6} {"to":<9} {"handled":<9} {"return":>7} {"best":>6}')
        for r in sweep:
            print(f'  {r["id"]:<4} {str(r["origin"]):<9} {r["direction"]:<6} {str(r["dest"]):<9} {r["handled"]:<9} '
                  f'{r["return"]:>7.0f} {r["best_after_shove"]:>6.0f}')

    print(f'\nRandom shove length (in a hurry): {draws} lengths per placement from '
          f'{shove_lengths(env, dist, mean, std, random.Random()).describe()}, '
          'redrawn until the shove fits in the room and is optimal')
    for agent in agents:
        sweep = [r for r in rows if r['agent'] == agent.name and r['shove'] == 'random-distance']
        if not sweep:
            continue
        best = sum(r['regret_after_shove'] == 0 for r in sweep)
        lengths = Counter(r['distance'] for r in sweep)
        by_length = {d: f'{sum(r["regret_after_shove"] == 0 for r in sweep if r["distance"] == d)}/{n}'
                     for d, n in sorted(lengths.items())}
        print(f'  {agent.name:<11} {len(sweep)} runs, best possible after the shove in {best}, mean length '
              f'{sum(r["distance"] for r in sweep) / len(sweep):.2f}; best-after-shove by length: {by_length}')

    validation = validate_shove_distance(env, mean, std, seed=seed, dist=dist)
    b = validation['bounded_sampling']
    print(f'\nShove length check ({"PASSED" if validation["passed"] else "FAILED"}): raw draws mean '
          f'{validation["raw_draws"]["mean"]}, std {validation["raw_draws"]["std"]} (expected {validation["expected_mean"]}, '
          f'{validation["expected_std"]}), counts {validation["raw_draws"]["counts"]}; from {b["placements"]} random '
          f'cells and directions, {b["accepted"]} shoves accepted (lengths {b["accepted_min"]}-{b["accepted_max"]}, '
          f'all landed in the room: {b["accepted_landed_in_room"] == b["accepted"]}), rejected draws '
          f'{b["rejected_draws_by_reason"]}, env refused all rejected: {b["rejected_but_env_allowed"] == 0}')

    if not args.no_save:
        out = save_results(rows, env, memory, args.demos, args.q_episodes, seed, args.results_dir, mean, std, draws,
                           validation, dist)
        print(f'\nSaved {len(rows)} runs to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}')
    print('Replay any run in the GUI, e.g.: uv run python gui.py --replay m6\n')


if __name__ == '__main__':
    main()
