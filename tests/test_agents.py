from collections import Counter
import random

import pytest

from agents import MemoryAgent, contains, segment_states
from experiment import random_shove_sweep, run_all, run_episode, save_results, shove_opportunities
from human import RandomShoveHuman, ScriptedHuman
from livingroom import Action, LivingRoom, Oracle, rollout


@pytest.fixture(scope='module')
def trained():
    random.seed(0)
    env = LivingRoom()
    memory = MemoryAgent(env)
    oracle = Oracle(env, rng=random.Random(0))  # one oracle, so tie-breaking varies across demos
    memory.train(rollout(env, oracle.act) for _ in range(200))
    return env, memory


def test_memory_groups_one_per_latent_setting(trained):
    env, memory = trained
    assert max(len(routes) for routes in memory.groups.values()) > 1  # many tied routes remembered
    returns = {rollout(env, Oracle(env).act, hurry=h).ret for h in (True, False)}
    assert set(memory.groups) == returns


def test_without_correction_follows_most_common_route(trained):
    env, memory = trained
    assert run_episode(env, memory, hurry=False, human=None)['return'] == env.optimal_return()
    assert run_episode(env, memory, hurry=True, human=None)['return'] < env.optimal_return()


@pytest.mark.parametrize('hurry', [False, True])
def test_one_correction_recovers_optimal_return(trained, hurry):
    env, memory = trained
    result = run_episode(env, memory, hurry=hurry, human=ScriptedHuman(env))
    assert result['corrections'] == (1 if hurry else 0)  # a rational human only shoves onto the rug in a hurry
    assert result['return'] == result['optimal']


def test_human_shoves_three_cells_once(trained):
    env, _ = trained
    env.reset(hurry=True)
    human = ScriptedHuman(env)
    assert human.intervene() == (Action.RIGHT, 3)
    assert human.intervene() is None


def test_shove_lands_on_rug_and_only_hurry_routes_contain_it(trained):
    env, memory = trained
    segment = segment_states(env.start, Action.RIGHT, (env.start[0], env.start[1] + 3))
    assert segment[-1] in env.rug
    hurry_ret = rollout(env, Oracle(env).act, hurry=True).ret
    assert {ret for ret, ts in memory.groups.items() if any(contains(t, segment) for t in ts)} == {hurry_ret}


def test_filter_keeps_only_routes_containing_the_correction(trained):
    env, memory = trained
    memory.reset()
    origin = env.start
    dest = (origin[0], origin[1] + 3)
    memory.observe_correction(origin, Action.RIGHT, 3, dest)
    segment = segment_states(origin, Action.RIGHT, dest)
    assert memory.candidate_routes()
    assert all(contains(t, segment) for t in memory.candidate_routes())


def test_shove_counts_steps_and_reward():
    env = LivingRoom()
    env.reset(hurry=True)
    pos, reward, _ = env.shove(Action.RIGHT, 2)
    assert pos == (env.start[0], env.start[1] + 2)
    assert env.steps == 2 and reward == 2 * (-1 - env.hurry_penalty)


def test_random_shoves_are_rational_and_three_cells(trained):
    env, memory = trained
    spots = shove_opportunities(env, memory, hurry=True)
    assert spots
    oracle = Oracle(env)
    for k, pos, directions in spots:
        for d in directions:
            result = run_episode(env, memory, True, RandomShoveHuman(env, at=k, direction=d))
            origin, direction, dest = result['shove']
            assert origin == pos and direction == d and result['corrections'] == 1
            assert abs(dest[0] - origin[0]) + abs(dest[1] - origin[1]) == 3


def test_no_random_shove_when_not_in_a_hurry(trained):
    env, memory = trained
    assert shove_opportunities(env, memory, hurry=False) == []


def test_sweep_reports_how_each_shove_was_handled(trained):
    env, memory = trained
    rows = random_shove_sweep(env, memory)
    assert {r['choice'] for r in rows} <= {'filter', 'toward'}
    assert all(r['return'] == r['best_after_shove'] for r in rows if r['choice'] == 'filter')


def test_every_random_shove_leads_to_best_return_after_the_shove(trained):
    env, memory = trained
    rows = random_shove_sweep(env, memory)
    assert rows and all(r['return'] == r['best_after_shove'] for r in rows)


def test_results_saved_per_seed_and_settings(trained, tmp_path):
    import csv
    import json

    env, memory = trained
    rows = run_all(env, [memory])
    out = save_results(rows, env, memory, demos=200, q_episodes=1, seed=7, root=tmp_path)
    assert out.parent == tmp_path and out.name == 'seed7_demos200_q1_shove-uniform1-6-draws5'
    with open(out / 'runs.csv') as f:
        saved = list(csv.DictReader(f))
    assert [r['id'] for r in saved] == [r['id'] for r in rows]
    assert float(saved[0]['return']) == rows[0]['return']
    settings = json.loads((out / 'settings.json').read_text())
    assert settings['seed'] == 7 and settings['demos'] == 200 and settings['env']['max_shove'] == 6
    assert settings['shove_distance']['distribution'] == 'uniform'
    normal = save_results(rows, env, memory, demos=200, q_episodes=1, seed=7, root=tmp_path, dist='normal')
    assert normal.name == 'seed7_demos200_q1_shove-mean3.0-std1.0-draws5'


def test_uniform_shove_distance_covers_every_length_equally():
    from human import ShoveDistance

    lengths = ShoveDistance(rng=random.Random(0), kind='uniform', low=1, high=6)
    draws = [lengths.draw() for _ in range(60000)]
    counts = Counter(draws)
    assert set(counts) == set(range(1, 7)) and all(abs(c - 10000) < 500 for c in counts.values())
    mean, std = lengths.expected()
    assert mean == 3.5 and abs(std - (35 / 12) ** 0.5) < 1e-9
    with pytest.raises(ValueError):
        ShoveDistance(kind='poisson')


def test_shove_distance_is_normal_around_three():
    from human import ShoveDistance

    lengths = ShoveDistance(rng=random.Random(0))
    draws = [lengths.draw() for _ in range(20000)]
    mean = sum(draws) / len(draws)
    std = (sum((d - mean) ** 2 for d in draws) / len(draws)) ** 0.5
    assert lengths.mean == 3.0 and abs(mean - 3.0) < 0.05 and abs(std - 1.0) < 0.1


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


def test_sampled_lengths_only_accept_shoves_inside_the_room():
    from human import ShoveDistance

    env = LivingRoom()
    lengths = ShoveDistance(rng=random.Random(1))
    pos = (10, 17)  # two cells from the right wall
    for _ in range(200):
        d = lengths.sample(lambda n: env.shove_problem(pos, Action.RIGHT, n))
        assert 1 <= d <= 2
    assert lengths.rejected['out_of_room'] > 0


def test_random_length_runs_are_valid_and_memory_reaches_best_after_shove(trained):
    from experiment import run_all

    env, memory = trained
    rows = [r for r in run_all(env, [memory]) if r['shove'] == 'random-distance']
    assert rows
    for r in rows:
        assert 1 <= r['distance'] <= env.max_shove and env.in_bounds(r['dest'])
        assert r['return'] == r['best_after_shove']


def test_validation_experiment_passes(trained):
    from experiment import validate_shove_distance

    env, _ = trained
    for dist in ('uniform', 'normal'):
        result = validate_shove_distance(env, n=2000, dist=dist)
        assert result['passed'], dist
    assert result['bounded_sampling']['rejected_but_env_allowed'] == 0
