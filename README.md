# causal_robot

Robots and RL agents often see several optimal trajectories in training. Which one is optimal depends on
latent variables the agent cannot observe, and some trajectories are rarer than others. A standard RL agent
optimises expected return, so it always follows the most commonly optimal trajectory. When a human corrects
it at deployment, it cannot tell which trajectory the correction points to.

We instead keep a **memory of every trajectory that can be optimal**. When the human corrects the robot, the
agent filters its memory to the optimal trajectories that contain that correction and follows one of them.
If no remembered trajectory contains the correction, the human is assumed to be rational and to observe the
latent variables. In that case the agent picks the highest-return trajectory among the valid options.

## Environment (`livingroom.py`)

A 20×20 room with a 12×14 rug in the middle that starts 3 cells right of the dock. The robot starts at the
dock and must reach the flag.

- **Latent variable `hurry`** (true 20% of the time, hidden from agents).
  - Not in a hurry: rug cells are costly, so going around the rug is optimal (return −31).
    Many tied routes exist, over or under the rug.
  - In a hurry: every step is costly, so the single straight line across the rug is optimal (return −57).
- **Human correction**: a shove of 1–`max_shove` cells (6) in one direction. The forced moves are ordinary
  steps that cost time and earn reward. `shove_problem` checks that a shove fits within the environment: at
  least 1 cell, at most `max_shove`, and landing inside the room. `shove` refuses one that doesn't.
- `Oracle` is the informed policy that reads `hurry`. Only data generation and the human use it.

## Agents (`agents.py`) and human (`human.py`)

- `QLearningAgent`: the standard RL baseline. It learns to always go around the rug and ignores corrections.
- `MemoryAgent`: stores every optimal trajectory from training and groups them by return, which stands in for
  a latent setting. With no correction it behaves like the baseline. After a shove it uses those prior
  trajectories to decide what is optimal from where it landed:
  1. it keeps only the remembered trajectories that contain every cell of the shove, and
  2. if none do, it scores each group by how much closer the shove moved the robot to that group's remembered
     routes (the shove's direction points towards the right routes), takes the highest-return group among
     the best-scoring ones, and rejoins its route nearest to where the robot landed.
- `ScriptedHuman`: sees `hurry` and **shoves once, at a fixed place: 3 cells right from the dock, onto the
  rug**. It only does this when the shove is optimal, which means only when the robot is in a hurry. No
  not-in-a-hurry route touches the rug, so the shove matches only the straight hurry route. Varying where
  and when the shove happens is covered below. In the GUI you choose the shove yourself (1–6 cells, once).
- `RandomShoveHuman`: same rational human, but it shoves at a chosen or random moment when the robot is
  about to make a suboptimal move, in any direction where all the forced steps are optimal. Shoves are 3 cells
  unless given a length.
- `ShoveDistance`: random shove lengths in whole cells, either uniform over 1–`max_shove` (1–6, every
  length equally likely; the default) or from Normal(mean, std), rounded. A length is redrawn until it is acceptable: within the environment's bounds and, for the rational human,
  optimal all the way (it would not push the robot past row 10, for example).

## Running

```bash
uv sync
uv run python experiment.py   # train both agents, run every experiment, save to results/ (~25s first time)
uv run python experiment.py --seed 0 1 2 3   # several seeds, each saved to its own folder
uv run python experiment.py --distance-draws 10   # more random-length shoves per placement (uniform 1-6)
uv run python experiment.py --shove-dist normal --shove-mean 4 --shove-std 1.5   # normal shove lengths instead
uv run python gui.py          # interactive: space steps, click to shove once, a switches agent
uv run python gui.py --demo   # plays the five scripted scenarios on a loop (--once --snapshots DIR saves PNGs)
uv run python gui.py --replay m6   # replay one experiment run by ID (see below)
uv run pytest
```

Example result (seed 0):

| agent      | hurry | human | return | optimal |
|------------|-------|-------|-------:|--------:|
| q-learning | True  | no    |    −93 |     −57 |
| q-learning | True  | shove |    −99 |     −57 |
| memory     | True  | no    |    −93 |     −57 |
| memory     | True  | shove |    −57 |     −57 |

### Random shove placement

`experiment.py` also runs one hurried episode for every place `RandomShoveHuman` could shove (seed 0):

Most shoves happen after the robot has already taken a few wrong steps, so the yardstick is the best return
still reachable from where the shove left it ("best after shove"):

| agent      | placements | reached best after shove | mean return | mean best after shove |
|------------|-----------:|-------------------------:|------------:|----------------------:|
| q-learning |          9 |                        0 |      −101.0 |                 −75.0 |
| memory     |          9 |                        9 |       −75.0 |                 −75.0 |

The memory agent reaches it for all 9 placements on seeds 0–3. Only the shove along row 10, (10,2)→(10,5),
lies on a remembered route, so the filter handles that one. The other 8 (sideways onto the rug, or back up
toward row 10) match no remembered route and go to the direction fallback.

The fallback first compared returns alone, and that picked the around-the-rug group (−31 beats −57) for all
8, so the robot went back around (mean −99). A literal direction match, preferring the group whose remembered
moves near the shove go the same way, got only 2 of 8 right. Upward shoves match neither group, and the
around-the-rug routes also move right along row 16. Scoring how much closer the shove moved the robot to
each group's routes gets all 8 right.

### Random shove length

For every placement above, `experiment.py` also runs 5 shoves whose lengths come from `ShoveDistance`,
uniform over 1–6 cells and redrawn until acceptable. In seeds 0–3:

| agent      | runs per seed | reached best after shove        | mean length |
|------------|--------------:|---------------------------------|------------:|
| q-learning |         45–55 | 0, 0, 4, 0                      |   3.22–3.29 |
| memory     |            45 | 45, 45, 45, 43                  |   2.91–3.56 |

Accepted lengths average under 3.5 because a rational human never shoves past the best path. An upward
shove from (13,2) can go at most 3 cells, back to row 10, so longer draws are redrawn there.

The memory agent's two misses are `m16` and `m17` in seed 3: 1- and 2-cell shoves right from the dock, to
(10,1) and (10,2), which stop short of the rug. Some around-the-rug routes also start by going right along
row 10, so the filter keeps both groups and picks the more common one (around the rug, −93). A shove of 3 or
more cells from the dock lands on the rug and is unambiguous.

Earlier runs used Normal(3, 1) (memory 179/180, one miss of the same kind) and Normal(5, 1) (180/180,
because dock shoves were almost always 3+ cells). Normal lengths cluster within about a cell of the mean;
near walls, where only 1–2 cells fit, a high mean rarely draws a short enough length, so the shove is skipped
more often. Uniform lengths avoid that. Their results are in the `shove-mean…` folders under `results/`.

**Validation.** `validate_shove_distance` checks the sampler and the bounds check, and its result is saved for
every seed. Results for seeds 0–3:

- The raw draws have mean 3.48–3.53 and std 1.70–1.73 (uniform 1–6 should give 3.5 and 1.71). Each length
  1–6 was drawn 792–874 times out of 5000, within 15% of an even split.
- 5000 random cells and directions are sampled with the bounds check alone. 4720–4748 shoves were accepted,
  with lengths 1–6. Every accepted shove landed inside the room, exactly that many cells away.
- Draws were rejected as `out_of_room` (uniform draws are never `too_short` or `too_long`), and the
  environment refused every one of those shoves. For normal lengths it also checks `too_short` (below 1)
  and `too_long` (above 6).

### Saved results

Each run of `experiment.py` saves to `results/seed<S>_demos<D>_q<Q>_shove-<lengths>-draws<N>/`, one folder
per seed and settings. `<lengths>` is `uniform1-6`, or `mean<M>-std<SD>` with `--shove-dist normal`:

- `runs.csv` has one row per run: ID, agent, `hurry`, shove type, placement and length, how the memory agent
  handled the shove (`filter` or `toward`), return, best possible return, best possible after the shove, and
  regrets.
- `settings.json` records the seed, demonstrations, Q-learning episodes, shove-length settings, environment,
  the memory agent's route groups, the git commit (marked `-dirty` if the code had uncommitted changes) and
  when it ran.
- `shove_distance_check.json` holds the validation of the random shove lengths described above.

Rerunning with the same settings overwrites that folder; the results are deterministic for a given commit.
Pass `--results-dir DIR` to save somewhere else, or `--no-save` to only print.

### Replaying runs in the GUI

`experiment.py` prints an ID for every run, `m…` for the memory agent and `q…` for Q-learning (seed 0):
`m1`–`m4` are the fixed-shove table, `m5`–`m13` the 3-cell random placements, and `m14` onward the
random-length shoves (likewise for `q`). Pass IDs to the GUI to watch those exact runs (the GUI replays
seed 0):

```bash
uv run python gui.py --replay m6          # one run
uv run python gui.py --replay m5 m6 q6    # several; n / p step between them, return replays
uv run python gui.py --replay m           # every memory-agent run (q: every Q-learning run, all: both)
```

The GUI trains its agents with the same seed and settings as `experiment.py`, and its caption shows the
return the experiment recorded next to the replayed one. `--once` plays the runs and exits, and reports any
replay whose return differs. `--snapshots DIR` saves PNGs at the start, shove and end of each run. The
learned Q-table is cached in `.cache/`, so replaying Q-learning runs does not retrain.

## Known limitations

- Grouping trajectories by return assumes that different latent settings never produce equal optimal
  returns. That holds here but not in general.
- A shove that lies on remembered routes from more than one group (such as a short shove along row 10
  near the dock) is resolved by group frequency, which picks the around-the-rug group.
- The direction fallback is tested on only two latent groups and one room. A shove that moves the robot
  equally close to several groups falls back to highest return, which compares returns from different
  reward functions.
