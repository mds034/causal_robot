# causal_robot

Robots and RL agents often see several optimal trajectories in training. Which one is optimal depends on
latent variables the agent cannot observe, and some trajectories are rarer than others. A standard RL agent
optimises expected return, so it follows the most commonly optimal trajectory. When a human corrects it at
deployment, it cannot tell which trajectory the correction points to.

Our agent learns with the **same RL algorithm** as the baseline, but also keeps a **replay buffer of its own
training experience, grouped by the rewards each episode paid**. It is never told about the latent variable,
or how many groups to expect. When the human corrects the robot, it follows the group of past experience
whose best moves agree with the correction.

## Environment (`livingroom.py`)

A 20×20 room with a 12×14 rug in the middle that starts 3 cells right of the dock. The robot starts at the
dock and must reach the flag.

- **Latent variable `hurry`** (true 20% of the time, hidden from agents).
  - Not in a hurry: rug cells are costly, so going around the rug is optimal (return −31).
    Many tied routes exist, over or under the rug.
  - In a hurry: every step is costly, so the single straight line across the rug is optimal (return −57).
- **Reward.** `LivingRoom(reward='dense')` (the default) pays each step's cost, `step_reward(pos)`, as it
  happens. `reward='sparse'` pays nothing until the episode ends, then the whole return at once; tabular
  Q-learning cannot learn the task that way (it never reaches the goal), so training uses dense reward. Agents'
  policies never take reward as input, so the reward cannot make their behaviour depend on `hurry`.
  `env.accrued` holds the return so far in both modes.
- **Human correction**: a shove of 1–`max_shove` cells (6) in one direction. The forced moves are ordinary
  steps that cost time and accrue reward. `shove_problem` checks that a shove fits within the environment: at
  least 1 cell, at most `max_shove`, and landing inside the room. `shove` refuses one that doesn't.
- **Observation.** Agents act on `env.observe()`, an `Obs(pos, on_rug, goal, rug)`: their position, whether
  they are on the rug, the goal's position and the rug's bounds. It never includes `hurry`. For the tabular
  agents these extra features change nothing (they are fixed or a function of position); they matter for an
  agent that generalises across states.
- `Oracle` is the informed policy that reads `hurry`. Only the evaluation (best possible return) and tests use
  it.

## Agents (`agents.py`)

- `QLearningAgent`: the standard RL baseline. It trains with online tabular Q-learning (ε-greedy) and keeps
  every transition it experienced, with how often it saw it. After training it **replays** that experience
  through the same Q-learning update until the values settle (`replay_values`), with each move's rewards
  averaged by how often each was seen, and acts on the result. It ignores corrections.

  Why replay: the online Q-table is only reliable along the route the agent travels. It explored the rug
  early, while everything else was still badly estimated, and then hardly ever again (under one update per
  rug cell–action pair per 1000 episodes), so its rug values stayed about 20 too optimistic with all four
  moves nearly tied. After a shove onto the rug it then preferred staying on the rug and bounced between two
  cells until `max_steps`. The replayed values do not have that problem.
- `MemoryAgent`: the same (it inherits `train`, so it learns an identical Q-table, keeps the same experience
  and replays it identically), so without a correction it acts exactly like the baseline. It also keeps its
  training experience grouped:
  1. **Grouping.** Each finished episode joins the group it agrees with on the most cells: same reward for
     entering the same cell, and no disagreement on any cell. If it agrees with none, it starts a new group.
     If it agrees with several groups that agree with each other, they merge. Nothing tells the agent what the
     groups mean; in this room they come out as exactly the two `hurry` regimes (purity 1.0 on every seed so
     far, checked by the experiment against labels the agent never sees).
  2. **Per-group values.** After training, each group's experience is replayed with the same
     `replay_values`, giving one Q-function per group. Cells a group never acted
     from are unusable for it.
  3. **Following a correction.** A shove is a run of (cell, direction) pairs. The agent follows the group for
     which every one of them is a greedy action (the most common such group if several are). If none is,
     `no-match`, and it keeps its Q policy.

  Why not store whole trajectories? Ranking a correction's matches by return favours the regime with cheaper
  steps (a not-hurried episode that stepped onto the rug and back scores −38, the hurried crossing −57), and
  ε-greedy almost never produces a whole hurried-optimal trajectory. Its single moves do all occur, and
  Q-learning stitches them together.

Neither agent trains with the human present. The only human model the memory agent assumes is that a
correction consists of moves that are best under some group of its past experience.

## Human (`human.py`)

`Human` sees `hurry` and shoves at most once per episode:

- **In a hurry:** at a step drawn from Normal(2, 1), rounded and clipped to the first 5 steps (0–4); always to
  the right; length uniform over 1–6. If the rug lies to the robot's right, the shove is lengthened to reach
  it (from the dock, 1–2 become 3).
- **Not in a hurry:** only while the robot is on the rug. A delay drawn from Normal(3, 1.5), clipped to 0–6
  steps, starts the first time the robot steps on the rug. When it has passed, the human pushes the robot off
  by the nearest rug edge (ties broken at random), just far enough to leave the rug.

## Running

```bash
uv sync
uv run python experiment.py   # train both agents, run every experiment, save to results/ (~2 min first time)
uv run python experiment.py --seed 0 1 2 3   # several seeds, each saved to its own folder
uv run python experiment.py --human-episodes 100   # more deployment episodes with the human per regime
uv run python gui.py          # interactive: space steps, click to shove once, a switches agent
uv run python gui.py --demo   # plays four scenarios on a loop (--once --snapshots DIR saves PNGs)
uv run python gui.py --replay m53   # replay one experiment run by ID (see below)
uv run pytest
```

## Results

60000 Q-learning episodes, 50 episodes with the human per agent and regime. Success means reaching the best
return still possible after the shove (or from the start, if there was no shove). Prior-weighted success is
0.8 × not hurried + 0.2 × hurried.

| seed | q-learning: not hurried | q-learning: hurried | q-learning: weighted | memory: not hurried | memory: hurried | memory: weighted |
|-----:|------:|------:|-----:|------:|------:|-----:|
|    0 |  1.00 |  0.00 | 0.80 |  1.00 |  1.00 | 1.00 |
|    1 |  1.00 |  0.00 | 0.80 |  1.00 |  1.00 | 1.00 |
|    2 |  1.00 |  0.06 | 0.81 |  1.00 |  1.00 | 1.00 |
|    3 |  1.00 |  0.00 | 0.80 |  1.00 |  1.00 | 1.00 |

- On every seed the memory agent found exactly 2 groups, each matching one value of `hurry` perfectly, and
  every hurried shove chose the hurried group (50/50 per seed).
- After a hurried shove the baseline leaves the rug and goes around: it reaches the goal every time, with a
  mean regret of 27–33 after the shove.
- Seed 2's 3 baseline successes are pushes of 6 cells from (10,1) to (10,7), deep in the rug on the dock's
  row. From there, crossing straight is the best route even averaged over `hurry`, so the baseline's choice
  coincides with the hurried optimum.
- With 30000 episodes, seed 2's Q-learning settled on a not-hurried route 2 steps longer than optimal; 60000
  is enough on all four seeds.

### Saved results

Each run of `experiment.py` saves to `results/seed<S>_q<Q>_human<N>/`, one folder per seed and settings:

- `runs.csv` has one row per run: ID, agent, `hurry`, whether the human was present, the step, direction and
  length of the shove, where it went, how the memory agent handled it (`filter` or `no-match`), return, best
  possible return, best possible after the shove, regrets, and whether the goal was reached.
- `settings.json` records the seed, Q-learning episodes, episodes with the human, the reward mode, the human
  model, the environment, the memory agent's groups (episodes, hurried and not-hurried counts, distinct
  transitions, purity), the git commit (marked `-dirty` if the code had uncommitted changes) and when it ran.

Rerunning with the same settings overwrites that folder; the results are deterministic for a given commit.
Pass `--results-dir DIR` to save somewhere else, or `--no-save` to only print.

### Replaying runs in the GUI

`experiment.py` prints an ID for every run, `m…` for the memory agent and `q…` for Q-learning. For each agent,
with the default 50 episodes: `1` is not hurried without the human, `2`–`51` not hurried with the human, `52`
hurried without the human, and `53`–`102` hurried with the human. Each run seeds its own human, so it replays
exactly. Pass IDs to the GUI to watch those runs (the GUI replays seed 0):

```bash
uv run python gui.py --replay m53         # one run
uv run python gui.py --replay m53 q53     # several; n / p step between them, return replays
uv run python gui.py --replay m           # every memory-agent run (q: every Q-learning run, all: both)
```

The GUI trains its agents with the same seed and settings as `experiment.py`, and its caption shows the
return the experiment recorded next to the replayed one. `--once` plays the runs and exits, and reports any
replay whose return differs. `--snapshots DIR` saves PNGs at the start, shove and end of each run. Trained
agents are cached in `.cache/`, so replaying does not retrain.

## Known limitations

- **Grouping assumes deterministic rewards:** two episodes disagree if they paid different rewards for the
  same cell. Noisy rewards would need a tolerance or a statistical test.
- **Deployment-only framing:** no human in training. The memory agent assumes a correction consists of moves
  that are best under some group of its experience, i.e. a human who acts optimally for the true conditions.
- The learner uses fixed ε and α; 30000 episodes was not enough to converge on seed 2, so the default is 60000.
