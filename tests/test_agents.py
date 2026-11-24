from collections import Counter
import random

import pytest

from agents import MemoryAgent, QLearningAgent, replay_values, segment_states
from experiment import group_report, run_all, run_episode, save_results, summarise, train
from human import Human
from livingroom import Action, LivingRoom, Obs, Oracle, Trajectory, rollout


# --- environment -------------------------------------------------------------

def test_shove_counts_steps_and_reward():
    env = LivingRoom()
    env.reset(hurry=True)
    pos, reward, _ = env.shove(Action.RIGHT, 2)
    assert pos == (env.start[0], env.start[1] + 2)
    assert env.steps == 2 and reward == env.accrued == 2 * (-1 - env.hurry_penalty)


def test_dense_reward_pays_each_step():
    env = LivingRoom()
    for hurry in (False, True):
        traj = rollout(env, Oracle(env).act, hurry=hurry)
        assert all(r == env.step_reward(s) for r, s in zip(traj.rewards, traj.states[1:]))
        assert sum(traj.rewards) == traj.ret == env.optimal_return()


def test_sparse_reward_is_paid_only_at_the_end():
    env = LivingRoom(reward='sparse')
    env.reset(hurry=False)
    oracle, rewards, done = Oracle(env), [], False
    while not done:
        _, reward, done, _ = env.step(oracle.act())
        rewards.append(reward)
    assert all(r == 0.0 for r in rewards[:-1])
    assert rewards[-1] == env.optimal_return() == env.accrued
    with pytest.raises(ValueError):
        LivingRoom(reward='weekly')


def test_env_rejects_shoves_outside_the_room():
    env = LivingRoom()
    env.reset(hurry=True)  # robot at the dock, on the left wall
    assert env.shove_problem(env.pos, Action.LEFT, 1) == 'out_of_room'
    assert env.shove_problem(env.pos, Action.RIGHT, 0) == 'too_short'
    assert env.shove_problem(env.pos, Action.RIGHT, env.max_shove + 1) == 'too_long'
    assert env.shove_problem(env.pos, Action.UP, 10) == 'too_long'
    with pytest.raises(ValueError):
        env.shove(Action.LEFT, 1)
    assert env.pos == env.start and env.steps == 0  # a rejected shove does not move the robot


def test_observation_has_position_rug_and_goal_but_not_hurry():
    env = LivingRoom()
    rug_rows, rug_cols = [r for r, _ in env.rug], [c for _, c in env.rug]
    for hurry in (False, True):
        env.reset(hurry)
        obs = env.observe()
        assert obs == Obs(env.start, False, env.goal, (min(rug_rows), min(rug_cols), max(rug_rows), max(rug_cols)))
        env.shove(Action.RIGHT, 3)
        assert env.observe().on_rug and env.observe().pos == (env.start[0], 3)
    assert 'hurry' not in Obs._fields



# --- human -------------------------------------------------------------------

def place(env: LivingRoom, pos, steps: int = 0) -> None:
    env.pos, env.steps = pos, steps


def first_shove(env: LivingRoom, human: Human, hurry: bool, act):
    # run one episode with policy `act`; return (step, origin, direction, distance, dest) of the shove, or None
    env.reset(hurry)
    human.reset()
    found = None
    while not env.done:
        shove = human.intervene()
        if shove:
            assert found is None, 'the human shoved twice'
            origin, step = env.pos, env.steps
            dest, _, _ = env.shove(*shove)
            found = (step, origin, *shove, dest)
            continue
        env.step(act(env.pos))
    return found


def test_hurried_shove_is_once_early_rightward_and_reaches_the_rug():
    env = LivingRoom()
    oracle = Oracle(env)
    for seed in range(200):
        human = Human(env, random.Random(seed))
        # a robot that walks down the left wall, as if going round the rug
        step, origin, direction, distance, dest = first_shove(
            env, human, True, lambda p: Action.DOWN if p[0] < env.size - 1 else oracle.act(p))
        assert 0 <= step <= Human.HURRY_LAST and direction == Action.RIGHT
        assert 1 <= distance <= env.max_shove and dest in env.rug


def test_hurried_shove_timing_and_distance_distribution():
    env = LivingRoom()
    env.reset(hurry=True)
    human = Human(env, random.Random(0))
    dues = []
    for _ in range(20000):
        human.reset()
        dues.append(human.due)
    assert set(dues) <= set(range(Human.HURRY_LAST + 1))
    assert abs(sum(dues) / len(dues) - Human.HURRY_MEAN) < 0.05
    # from the dock (3 cells from the rug), lengths 1-2 are raised to 3; 4-6 are kept
    counts = Counter()
    for _ in range(6000):
        env.reset(hurry=True)
        human.reset()
        human.due = 0
        counts[human.intervene()[1]] += 1
    assert set(counts) == {3, 4, 5, 6} and abs(counts[3] - 3000) < 200


def test_no_calm_shove_off_the_rug():
    env = LivingRoom()
    oracle = Oracle(env)
    for seed in range(50):
        assert first_shove(env, Human(env, random.Random(seed)), False, oracle.act) is None


def test_calm_shove_pushes_just_off_the_rug_by_the_nearest_edge_after_the_delay():
    env = LivingRoom()
    rug_rows = [r for r, _ in env.rug]
    for seed in range(100):
        env.reset(hurry=False)
        human = Human(env, random.Random(seed))
        human.reset()
        place(env, (min(rug_rows) + 1, 10), steps=5)  # 2 cells below the top edge, far from the others
        for t in range(human.due):
            place(env, env.pos, steps=5 + t)
            assert human.intervene() is None
        place(env, env.pos, steps=5 + human.due)
        assert human.intervene() == (Action.UP, 2)
        dest, _, _ = env.shove(Action.UP, 2)
        assert dest not in env.rug and env.move(dest, Action.DOWN) in env.rug


def test_calm_shove_breaks_ties_at_random():
    env = LivingRoom()
    rug_rows, rug_cols = [r for r, _ in env.rug], [c for _, c in env.rug]
    corner = (min(rug_rows) + 1, min(rug_cols) + 1)  # 2 cells from both the top and the left edge
    directions = Counter()
    for seed in range(400):
        env.reset(hurry=False)
        human = Human(env, random.Random(seed))
        human.reset()
        human.due = 0
        place(env, corner)
        directions[human.intervene()] += 1
    assert set(directions) == {(Action.UP, 2), (Action.LEFT, 2)}
    assert abs(directions[Action.UP, 2] - 200) < 50


# --- agents ------------------------------------------------------------------

def test_memory_agent_learns_the_same_q_table_as_the_baseline():
    env = LivingRoom()
    tables = []
    for cls in (QLearningAgent, MemoryAgent):
        random.seed(3)
        agent = cls(env, seed=3)
        agent.train(300)
        tables.append((agent.q, agent.experience, agent.values))
    assert tables[0] == tables[1]  # same online Q-table, same experience, same replayed values


def test_replay_averages_rewards_by_how_often_they_were_seen():
    goal, s = (0, 1), (0, 0)
    experience = Counter({(s, Action.RIGHT, -6.0, goal): 4, (s, Action.RIGHT, -3.0, goal): 1})
    assert abs(replay_values(experience, 0.1, goal)[s, Action.RIGHT] - (-5.4)) < 1e-4


def test_replay_marks_moves_into_unknown_cells_unusable():
    goal = (0, 2)
    experience = Counter({((0, 0), Action.RIGHT, -1.0, (0, 1)): 1, ((0, 0), Action.DOWN, -1.0, (1, 0)): 1,
                          ((0, 1), Action.RIGHT, -1.0, goal): 1})
    q = replay_values(experience, 0.1, goal)
    assert q[(0, 0), Action.DOWN] == float('-inf') and abs(q[(0, 0), Action.RIGHT] - (-2.0)) < 1e-4


def _walk(env, hurry, rng, steps):
    env.reset(hurry)
    states, actions, rewards = [env.pos], [], []
    for _ in range(steps):
        a = rng.choice(list(Action))
        pos, r, done, _ = env.step(a)
        states.append(pos), actions.append(a), rewards.append(r)
        if done:
            break
    return Trajectory(tuple(states), tuple(actions), env.accrued, tuple(rewards))


def test_episodes_are_grouped_by_the_rewards_they_paid():
    # hand-built experience: oracle routes and random walks from both regimes, interleaved
    env = LivingRoom()
    rng = random.Random(0)
    memory = MemoryAgent(env)
    truth = []
    for i in range(40):
        hurry = i % 3 == 0
        traj = rollout(env, Oracle(env).act, hurry=hurry) if i % 2 else _walk(env, hurry, rng, 80)
        memory.remember(traj)
        truth.append(hurry)
    assert len(memory.groups) == 2
    for k in range(2):
        assert len({h for h, g in zip(truth, memory.episode_groups) if g == k}) == 1  # each group is one regime


def test_a_conflicting_episode_starts_a_new_group_and_an_agreeing_one_joins():
    env = LivingRoom()
    memory = MemoryAgent(env)
    a, b = (env.start[0], 1), (env.start[0], 2)
    memory.remember(Trajectory((env.start, a, b), (Action.RIGHT, Action.RIGHT), -2.0, (-1.0, -1.0)))
    memory.remember(Trajectory((env.start, a), (Action.RIGHT,), -3.0, (-3.0,)))  # same cell, other reward
    memory.remember(Trajectory((env.start, a), (Action.RIGHT,), -1.0, (-1.0,)))  # agrees with the first
    assert memory.episode_groups == [0, 1, 0]


def test_groups_that_never_overlapped_merge_when_an_episode_bridges_them():
    env = LivingRoom()
    memory = MemoryAgent(env)
    up, down = (env.start[0] - 1, 0), (env.start[0] + 1, 0)
    memory.remember(Trajectory((env.start, up), (Action.UP,), -3.0, (-3.0,)))
    memory.remember(Trajectory((env.start, (env.start[0], 1)), (Action.RIGHT,), -1.0, (-1.0,)))  # other regime
    memory.remember(Trajectory((env.start, down), (Action.DOWN,), -3.0, (-3.0,)))  # overlaps neither
    assert len(memory.groups) == 3
    bridge = (env.start, up, env.start, down)  # enters both cells, agreeing with the first and third episodes
    memory.remember(Trajectory(bridge, (Action.UP, Action.DOWN, Action.DOWN), -9.0, (-3.0, -3.0, -3.0)))
    assert len(memory.groups) == 2 and memory.episode_groups == [0, 1, 0, 0]
    assert memory.groups[0].episodes == 3


@pytest.fixture(scope='module')
def trained():
    env = LivingRoom()
    memory = train('memory', env, q_episodes=5000, seed=0, cache=False)
    return env, memory


def test_grouping_recovers_the_hidden_regimes(trained):
    env, memory = trained
    report = group_report(memory)
    assert len(report) == 2 and all(g['purity'] == 1.0 for g in report)
    assert sorted(g['hurried_episodes'] > 0 for g in report) == [False, True]


def hurried_group(memory):
    return next(g['group'] for g in group_report(memory) if g['hurried_episodes'])


def test_a_hurried_shove_agrees_only_with_the_hurried_group_and_is_followed_optimally(trained):
    env, memory = trained
    env.reset(hurry=True)
    memory.reset()
    origin, dest = env.start, (env.start[0], 3)
    assert memory.consistent_groups(origin, Action.RIGHT, dest) == [hurried_group(memory)]
    env.shove(Action.RIGHT, 3)
    memory.observe_correction(origin, Action.RIGHT, 3, dest)
    assert memory.last_choice == 'filter' and memory.chosen == hurried_group(memory)
    assert memory.candidate_routes() and memory.candidate_routes()[0].states[-1] == env.goal
    while not env.done:
        env.step(memory.act(env.observe()))
    assert env.pos == env.goal and env.accrued == env.optimal_return()


def test_without_a_correction_the_memory_agent_acts_like_the_baseline(trained):
    env, memory = trained
    memory.reset()
    for r in range(env.size):
        for c in range(env.size):
            env.pos = (r, c)
            obs = env.observe()
            assert memory.act(obs) == QLearningAgent.act(memory, obs)


def test_the_baseline_reaches_the_goal_from_every_rug_cell(trained):
    # replayed values do not loop where the online Q-table, rarely updated off its route, did
    env, memory = trained
    for cell in sorted(env.rug):
        env.reset(hurry=False)
        env.pos = cell
        while not env.done:
            env.step(QLearningAgent.act(memory, env.observe()))
        assert env.pos == env.goal, cell


def test_an_inconsistent_shove_falls_back_to_the_baseline(trained):
    env, memory = trained
    memory.reset()
    memory.observe_correction((0, 5), Action.LEFT, 3, (0, 2))  # away from the goal: greedy in no group
    assert memory.last_choice == 'no-match' and memory.chosen is None and memory.candidate_routes() == []
    env.pos = (5, 5)
    assert memory.act(env.observe()) == QLearningAgent.act(memory, env.observe())


def test_segment_states():
    assert segment_states((0, 0), Action.RIGHT, (0, 2)) == [(0, 0), (0, 1), (0, 2)]


# --- experiment --------------------------------------------------------------

def test_with_the_human_memory_follows_hurried_shoves_and_baseline_does_not(trained):
    env, memory = trained
    rows = run_all(env, [memory], episodes=10)
    hurried = [r for r in rows if r['hurry'] and r['shove'] == 'human']
    assert hurried and all(r['direction'] == 'RIGHT' and 0 <= r['shove_step'] <= Human.HURRY_LAST for r in hurried)
    assert all(tuple(r['dest']) in env.rug for r in hurried)
    s = summarise(rows)
    assert s[True]['shoved'] == 1.0 and s[False]['shoved'] == 0.0
    assert s[True]['handled'] == {'filter': 10} and s[True]['success'] == 1.0 and s[False]['success'] == 1.0


def test_run_episode_reports_best_return_after_the_shove(trained):
    env, memory = trained

    class ShoveRightThree:
        def reset(self):
            self.used = False

        def intervene(self, intended=None):
            if self.used:
                return None
            self.used = True
            return Action.RIGHT, 3

    result = run_episode(env, memory, hurry=True, human=ShoveRightThree())
    assert result['shove_step'] == 0 and result['choice'] == 'filter'
    assert result['return'] == result['best_after_shove'] == result['optimal']


def test_results_saved_per_seed_and_settings(trained, tmp_path):
    import csv
    import json

    env, memory = trained
    rows = run_all(env, [memory], episodes=2)
    out = save_results(rows, env, memory, q_episodes=1, seed=7, root=tmp_path, episodes=2)
    assert out.parent == tmp_path and out.name == 'seed7_q1_human2'
    with open(out / 'runs.csv') as f:
        saved = list(csv.DictReader(f))
    assert [r['id'] for r in saved] == [r['id'] for r in rows]
    assert float(saved[0]['return']) == rows[0]['return']
    settings = json.loads((out / 'settings.json').read_text())
    assert settings['seed'] == 7 and settings['env']['max_shove'] == 6 and settings['reward'] == 'dense'
    assert len(settings['memory_groups']) == 2 and all(g['purity'] == 1.0 for g in settings['memory_groups'])
    assert 'hurry' in settings['human']
