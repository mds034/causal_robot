# What We Have/Should Have!

1. An environment (livingroom.py)
    - livingroom navigation task
    - hidden variable "is the human in a hurry" (0.8 no, 0.2 yes)
    - human can push robot as correction, based on hurry
    - observation: position, on-rug flag, goal position, rug bounds (never hurry)
    - reward: dense (per step) for training; sparse (all at the end) still available via reward='sparse'

2. Agents (agents.py)
    - Q-Learning baseline TODO PPO!!!!!!!
        - online tabular Q-learning, then replays all its experience to convergence (replay_values)
    - Memory Agent (ours)
        - same Q-learning and same replay, so without a correction it IS the baseline
        - replay buffer grouped by the rewards each episode paid (never told about hurry or how many groups)
        - one Q-function per group, from replaying that group's experience
        - correction filter based on specific (s, a) pairs: follow the group under which every pushed
          (cell, direction) is a best move

3. Experiments (experiment.py)
    - runs (main)
    - experimental details
        - different seeds (0-3), 60000 training episodes, 50 episodes with the human per agent and regime
        - grouping purity checked against the true hurry (the agent never sees it)

4. GUI (gui.py)
    - just the gui
    - loads the cached seed-0 agents; a switches agent, h toggles hurry, space steps, click to shove

5. Human model (human.py)
    - hard coded shoving policy

        HUMAN MODEL
        - at most one shove per episode

        1. if human IS in a hurry
            - SAMPLING LOCATION/TIMING: Normal(2, 1) over the first 5 steps (0-4)
            - SAMPLING DISTANCE: uniform over 1-6, raised so the shove reaches the rug
            - SAMPLING DIRECTION: always rightward

        2. if human is NOT in a hurry
            - SAMPLING LOCATION/TIMING: once on the rug, after a delay of Normal(3, 1.5) over 7 steps (0-6)
            - SAMPLING DISTANCE: minimum to get off the rug if possible, else max push
            - SAMPLING DIRECTION: to closest end of rug (ties broken 50/50)

# What We Need To Do!

0. ~~implement sparse reward function~~ done, then switched back to dense reward for training
   (Q-learning can't learn from the sparse one)

1. ~~implement the human model as we decided~~ done

2. move starting position up one

3. rerun experiments with new human model ~~and new starting position~~ (start not moved yet)

4. ~~confirm results: baseline 80%, ours 100%~~ confirmed on seeds 0-3 with the current start

...later:
1. implement PPO as an additional baseline
2. think about continuous action spaces and long horizon tasks
