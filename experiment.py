"""Train both agents with the same Q-learning (no human present), then deploy them with the human.

Every run gets an ID (m1, m2, ... for the memory agent, q1, q2, ... for Q-learning) that `gui.py --replay`
accepts, so any row printed here can be watched in the GUI. Each run seeds its own human, so it replays
identically.

Results are saved to results/seed<S>_q<Q>_human<N>/: runs.csv has one row per run, settings.json the
settings, environment, human model, memory agent's groups (with their purity) and git commit they came from. Rerunning with the same
settings overwrites them.
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
from human import Human
from livingroom import LivingRoom

Q_EPISODES, SEED, HUMAN_EPISODES = 60000, 0, 50
P_HURRY = 0.2  # the env's prior on `hurry`, used to weight the two regimes
CACHE_DIR = Path(__file__).parent / '.cache'
RESULTS_DIR = Path(__file__).parent / 'results'
COLUMNS = ('id', 'agent', 'hurry', 'shove', 'shove_step', 'direction', 'distance', 'origin', 'dest', 'handled',
           'return', 'optimal', 'regret', 'best_after_shove', 'regret_after_shove', 'reached_goal')
PREFIX = {'memory': 'm', 'q-learning': 'q'}
AGENT_CLASSES = {'memory': MemoryAgent, 'q-learning': QLearningAgent}


AGENT_STATE = ('q', 'experience', 'values', 'training_hurry')  # what is cached for every agent
MEMORY_STATE = ('groups', 'episode_groups')  # cached in addition for the memory agent


def train(name: str, env: LivingRoom, q_episodes: int = Q_EPISODES, seed: int = SEED, cache: bool = True):
    # Both agents run the same Q-learning with the same seed, so they learn the same Q table and replay the same
    # experience; the memory agent also keeps it grouped. Training takes about a minute, so the result is cached
    # per (agent, env, episodes, seed). The harness records each training episode's true `hurry` (agents never see it) to check the
    # memory agent's grouping afterwards.
    config = repr((name, env.reward_mode, 'obs-v1', 'groups-v1', 'replay-v1', env.size, env.start, env.goal, sorted(env.rug),
                   env.rug_penalty, env.hurry_penalty, env.max_steps, q_episodes, seed))
    path = CACHE_DIR / f'{name}-{hashlib.sha1(config.encode()).hexdigest()[:12]}.pkl'
    agent = AGENT_CLASSES[name](env, seed=seed)
    if cache and path.exists():
        agent.__dict__.update(pickle.loads(path.read_bytes()))
        env.reset()
        return agent
    random.seed(seed)  # the env samples `hurry` from the global RNG
    labels, reset = [], env.reset

    def recording_reset(hurry=None):
        pos = reset(hurry)
        labels.append(env.hurry)
        return pos

    env.reset = recording_reset
    try:
        agent.train(q_episodes)
    finally:
        del env.reset  # back to the class method
    agent.training_hurry = labels
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        keys = AGENT_STATE + (MEMORY_STATE if isinstance(agent, MemoryAgent) else ())
        state = {k: getattr(agent, k) for k in keys}
        path.write_bytes(pickle.dumps(state))
    return agent


def group_report(memory: MemoryAgent) -> list[dict]:
    # per group: its size, experience, and how pure it is with respect to the true `hurry` (which the agent never saw)
    labels = getattr(memory, 'training_hurry', [])
    out = []
    for i, g in enumerate(memory.groups):
        mine = [h for h, k in zip(labels, memory.episode_groups) if k == i]
        hurried = sum(mine)
        out.append({'group': i, 'episodes': g.episodes, 'transitions': len(g.transitions),
                    'hurried_episodes': hurried, 'calm_episodes': len(mine) - hurried,
                    'purity': max(hurried, len(mine) - hurried) / len(mine) if mine else None})
    return out


@dataclass(frozen=True)
class Run:
    """One deployment episode: which agent, the true `hurry`, and whether the human is present."""

    id: str
    agent: str  # 'memory' or 'q-learning'
    hurry: bool
    shove: str  # 'none' (no human) or 'human'
    seed: int = SEED

    def human(self, env: LivingRoom) -> Optional[Human]:
        return Human(env, random.Random(f'{self.seed}-{self.id}')) if self.shove == 'human' else None

    def describe(self) -> str:
        hurry = 'in a hurry' if self.hurry else 'not in a hurry'
        return f'{self.id}: {self.agent}, {hurry}, {"with the human" if self.shove == "human" else "no human"}'


def plan_runs(env: LivingRoom, agent, seed: int = SEED, episodes: int = HUMAN_EPISODES) -> list[Run]:
    # per regime: one run without the human, then `episodes` runs with it (each with its own seeded human)
    prefix, runs = PREFIX[agent.name], []
    for hurry in (False, True):
        for shove in ['none'] + ['human'] * episodes:
            runs.append(Run(f'{prefix}{len(runs) + 1}', agent.name, hurry, shove, seed))
    return runs


def run_episode(env: LivingRoom, agent, hurry: bool, human: Optional[Human] = None) -> dict:
    env.reset(hurry)
    agent.reset()
    if human:
        human.reset()
    corrections, shove, shove_step, best_after = 0, None, None, None
    while not env.done:
        intended = agent.act(env.observe())
        correction = human.intervene(intended) if human else None
        if correction:
            origin, shove_step = env.pos, env.steps
            dest, _, _ = env.shove(*correction)
            corrections += 1
            shove = (origin, correction[0], dest)
            best_after = env.accrued - env._cost[dest]  # return accrued so far plus the best possible from dest
            agent.observe_correction(origin, *correction, dest)
            continue
        env.step(intended)
    return {'return': env.accrued, 'optimal': env.optimal_return(), 'corrections': corrections,
            'reached_goal': env.pos == env.goal, 'shove': shove, 'shove_step': shove_step,
            'best_after_shove': best_after, 'choice': getattr(agent, 'last_choice', '-') if corrections else '-'}


def replay(env: LivingRoom, agent, run: Run) -> dict:
    return run_episode(env, agent, run.hurry, run.human(env))


def run_all(env: LivingRoom, agents, seed: int = SEED, episodes: int = HUMAN_EPISODES) -> list[dict]:
    # replay every planned run once and flatten it into a row with the COLUMNS above
    rows = []
    for agent in agents:
        for run in plan_runs(env, agent, seed, episodes):
            r = replay(env, agent, run)
            origin, direction, dest = r['shove'] or (None, None, None)
            best = r['best_after_shove']
            rows.append({
                'id': run.id, 'agent': run.agent, 'hurry': run.hurry, 'shove': run.shove,
                'shove_step': r['shove_step'], 'direction': direction.name if direction is not None else None,
                'distance': abs(dest[0] - origin[0]) + abs(dest[1] - origin[1]) if r['shove'] else None,
                'origin': origin, 'dest': dest, 'handled': r['choice'], 'return': r['return'],
                'optimal': r['optimal'], 'regret': r['optimal'] - r['return'], 'best_after_shove': best,
                'regret_after_shove': None if best is None else best - r['return'], 'reached_goal': r['reached_goal'],
            })
    return rows


def success(row: dict) -> bool:
    # the best return still possible: after the shove if there was one, otherwise from the start
    return (row['regret_after_shove'] if row['regret_after_shove'] is not None else row['regret']) == 0


def summarise(rows: list[dict]) -> dict:
    # per regime, over the runs with the human present
    out = {}
    for hurry in (False, True):
        sel = [r for r in rows if r['hurry'] == hurry and r['shove'] == 'human']
        n = len(sel)
        out[hurry] = {
            'runs': n,
            'success': sum(map(success, sel)) / n if n else 0.0,
            'shoved': sum(r['dest'] is not None for r in sel) / n if n else 0.0,
            'reached_goal': sum(r['reached_goal'] for r in sel) / n if n else 0.0,
            'mean_regret': sum(r['regret_after_shove'] if r['regret_after_shove'] is not None else r['regret']
                               for r in sel) / n if n else 0.0,
            'handled': dict(Counter(r['handled'] for r in sel if r['handled'] != '-')),
        }
    out['weighted_success'] = (1 - P_HURRY) * out[False]['success'] + P_HURRY * out[True]['success']
    return out


def results_folder(root: Path, seed: int, q_episodes: int, episodes: int) -> Path:
    return root / f'seed{seed}_q{q_episodes}_human{episodes}'


def save_results(rows: list[dict], env: LivingRoom, memory: MemoryAgent, q_episodes: int, seed: int,
                 root: Path = RESULTS_DIR, episodes: int = HUMAN_EPISODES) -> Path:
    out = results_folder(root, seed, q_episodes, episodes)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'runs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k: '' if v is None else v for k, v in row.items()} for row in rows)
    rug_rows, rug_cols = [r for r, _ in env.rug], [c for _, c in env.rug]
    settings = {
        'seed': seed, 'q_episodes': q_episodes, 'human_episodes_per_regime': episodes, 'reward': env.reward_mode,
        'human': Human.settings(),
        'env': {'size': env.size, 'start': env.start, 'goal': env.goal,
                'rug_rows': [min(rug_rows), max(rug_rows)], 'rug_cols': [min(rug_cols), max(rug_cols)],
                'rug_penalty': env.rug_penalty, 'hurry_penalty': env.hurry_penalty, 'max_steps': env.max_steps,
                'max_shove': env.max_shove},
        'memory_groups': group_report(memory),
        'git_commit': git_commit(),
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    (out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
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
    parser.add_argument('--q-episodes', type=int, default=Q_EPISODES, help='Q-learning training episodes (both agents)')
    parser.add_argument('--seed', type=int, nargs='+', default=[SEED], help='one or more seeds to run')
    parser.add_argument('--human-episodes', type=int, default=HUMAN_EPISODES,
                        help='deployment episodes with the human, per agent and per value of hurry')
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR, help='where to save results')
    parser.add_argument('--no-save', action='store_true', help='print results without saving them')
    args = parser.parse_args()
    for seed in args.seed:
        run_seed(seed, args)


def run_seed(seed: int, args: argparse.Namespace) -> None:
    env = LivingRoom()
    agents = [train(name, env, args.q_episodes, seed) for name in ('q-learning', 'memory')]
    memory = agents[1]
    rows = run_all(env, agents, seed, args.human_episodes)
    print(f'=== seed {seed}\nmemory agent: {len(memory.groups)} groups of training experience '
          '(purity checked against the true hurry, which the agent never saw):')
    for g in group_report(memory):
        purity = 'n/a' if g['purity'] is None else f'{g["purity"]:.3f}'
        print(f'  group {g["group"]}: {g["episodes"]} episodes ({g["hurried_episodes"]} hurried, '
              f'{g["calm_episodes"]} not), {g["transitions"]} distinct transitions, purity {purity}')

    print(f'\nWithout the human:\n{"id":<5} {"agent":<11} {"hurry":<6} {"return":>7} {"optimal":>8} '
          f'{"regret":>7} {"goal":>5}')
    for r in rows:
        if r['shove'] == 'none':
            print(f'{r["id"]:<5} {r["agent"]:<11} {r["hurry"]!s:<6} {r["return"]:>7.0f} {r["optimal"]:>8.0f} '
                  f'{r["regret"]:>7.0f} {r["reached_goal"]!s:>5}')

    print(f'\nWith the human ({args.human_episodes} episodes per agent and regime). Success = best return still '
          'possible after the shove (or from the start, if no shove):')
    for agent in agents:
        s = summarise([r for r in rows if r['agent'] == agent.name])
        for hurry in (False, True):
            t = s[hurry]
            print(f'  {agent.name:<11} hurry={hurry!s:<5} success {t["success"]:.2f}  shoved {t["shoved"]:.2f}  '
                  f'reached goal {t["reached_goal"]:.2f}  mean regret {t["mean_regret"]:.1f}  handled {t["handled"]}')
        print(f'  {agent.name:<11} prior-weighted success ({1 - P_HURRY:.1f} x not hurried + {P_HURRY:.1f} x hurried): '
              f'{s["weighted_success"]:.2f}')

    if not args.no_save:
        out = save_results(rows, env, memory, args.q_episodes, seed, args.results_dir, args.human_episodes)
        print(f'\nSaved {len(rows)} runs to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}')
    first_hurried = f'{PREFIX[memory.name]}{args.human_episodes + 3}'  # after the calm runs and the hurried no-human run
    print(f'Replay any run in the GUI, e.g. the first hurried one with the human: '
          f'uv run python gui.py --replay {first_hurried}\n')


if __name__ == '__main__':
    main()
