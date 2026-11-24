"""Tkinter viewer for LivingRoom: step an agent that cannot see `hurry`, and shove it once as a human would."""
from __future__ import annotations

import argparse
import math
import random
import subprocess
import tkinter as tk
from pathlib import Path
from typing import Callable, Iterator, Optional

import experiment
from experiment import PREFIX, Run, plan_runs
from human import Human
from livingroom import MOVES, Action, LivingRoom, Obs, Oracle, Pos

CELL = 34  # the artwork below is sized for this cell size
FRAME_MS = 16
LIFT = 12  # px the hand lifts the robot during a shove
COLORS = {
    'floor': '#eadcc2', 'grid': '#dccbab', 'shadow': '#bfae8e', 'trail': '#7f9cc7', 'shove': '#8e44ad',
    'rug': '#a8432f', 'rug_border': '#e8c07a', 'rug_dark': '#6e2619', 'rug_accent': '#2f5d7a',
    'robot': '#4a78c2', 'robot_hurry': '#e0662f', 'robot_head': '#d9dee6', 'robot_dark': '#2b2f3a',
    'light': '#ffd166', 'dock': '#9aa5b1', 'memory': '#4f9a5a', 'banner': '#f7efe0', 'skin': '#f2c29b', 'skin_dark': '#c98f6b', 'sleeve': '#6c5ce7',
}
REPLAY_HELP = 'n: next run   p: previous run   return: replay this run'
HELP = 'space: agent step   click a highlighted cell: shove (once)   a: switch agent   h: toggle hurry   r: reset'
AGENTS = ('memory', 'q-learning', 'oracle')
MAX_ROUTES_DRAWN = 40
DEMO_STEP_MS, DEMO_PAUSE_MS = 120, 2500
DEMO = (  # agent, hurry, human present?, caption
    ('memory', False, True, 'Not in a hurry, human present. The human only shoves a robot that is on the rug, '
                            'so the memory agent acts exactly like Q-learning.'),
    ('q-learning', True, True, 'Q-learning, in a hurry. Early on, the human shoves it right, far enough to '
                               'reach the rug. It has no memory to consult.'),
    ('memory', True, True, 'Memory agent, in a hurry. After the shove it follows the group of past experience '
                           'the shove agrees with (green: that group\'s route from here).'),
    ('memory', True, False, 'Memory agent, in a hurry, no human. Without a correction it acts exactly like '
                            'Q-learning.'),
)
Frame = tuple[float, float, float, float]  # robot x, robot y, lift, hand y


class OracleAgent:
    """Reference agent that cheats by reading `hurry`."""

    name = 'oracle'

    def __init__(self, env: LivingRoom):
        self.oracle = Oracle(env)

    def reset(self) -> None:
        pass

    def act(self, obs: Obs) -> Action:
        return self.oracle.act(obs.pos)

    def observe_correction(self, *_) -> None:
        pass

    def candidate_routes(self) -> list:
        return []


def ease(n: int) -> list[float]:
    # n cosine-eased steps from 0 (exclusive) to 1 (inclusive)
    return [(1 - math.cos(math.pi * i / n)) / 2 for i in range(1, n + 1)]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class LivingRoomGUI:
    def __init__(self, env: LivingRoom):
        self.env = env
        self.hover: Optional[Pos] = None
        self.animating = False
        self.agents: dict[str, object] = {}
        self.agent_name = AGENTS[0]
        self.interactive = True
        self.caption = ''
        self.snapshot_dir: Optional[Path] = None
        self.label = ''  # names snapshot files
        self.help = HELP
        self.on_episode_end: Callable[[], None] = lambda: None

        self.root = tk.Tk()
        self.root.title('LivingRoom')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=env.size * CELL, height=env.size * CELL, highlightthickness=0)
        self.canvas.pack()
        self.status = tk.Label(self.root, font=('Menlo', 12), justify='left', anchor='w')
        self.status.pack(fill='x', padx=8, pady=6)

        self.root.bind('<space>', lambda _: self.policy_step())
        self.root.bind('h', lambda _: self.reset(not self.env.hurry))
        self.root.bind('r', lambda _: self.reset())
        self.root.bind('a', lambda _: self.switch_agent())
        self.canvas.bind('<Button-1>', lambda e: self.shove_to(self.cell_at(e)))
        self.canvas.bind('<Motion>', lambda e: self.set_hover(self.cell_at(e)))
        self.canvas.bind('<Leave>', lambda _: self.set_hover(None))

        self.draw_room()
        self.reset()

    # --- interaction -------------------------------------------------------

    @property
    def agent(self):
        if self.agent_name not in self.agents:
            self.agents[self.agent_name] = self.train(self.agent_name)
        return self.agents[self.agent_name]

    def train(self, name: str):
        # agents are trained on demand; training samples `hurry` and leaves the env reset, so callers reset after
        self.status['text'] = f'training {name} agent...'
        self.root.update()
        if name == 'oracle':
            return OracleAgent(self.env)
        return experiment.train(name, self.env)  # same seed and settings as experiment.py, so runs match

    def switch_agent(self) -> None:
        if self.animating:
            return
        self.agent_name = AGENTS[(AGENTS.index(self.agent_name) + 1) % len(AGENTS)]
        self.reset(self.env.hurry)

    def reset(self, hurry: Optional[bool] = None) -> None:
        if self.animating:
            return
        agent = self.agent  # train first: training resets the env
        self.env.reset(hurry)
        agent.reset()
        self.trail, self.shoves, self.total = [self.env.pos], [], 0.0
        self.draw()

    def policy_step(self) -> None:
        if self.animating or self.env.done:
            return
        pos, reward, _, _ = self.env.step(self.agent.act(self.env.observe()))
        self.trail.append(pos)
        self.total += reward
        self.draw()

    def shove_targets(self) -> dict[Pos, tuple[Action, int]]:
        # cells reachable by a shove: up to `max_shove` cells in a straight line from the robot
        (r, c), targets = self.env.pos, {}
        for action, (dr, dc) in MOVES.items():
            for dist in range(1, self.env.max_shove + 1):
                cell = (r + dr * dist, c + dc * dist)
                if not self.env.in_bounds(cell):
                    break
                targets[cell] = (action, dist)
        return targets

    def shove_to(self, cell: Pos) -> None:
        target = self.shove_targets().get(cell)
        if self.animating or self.env.done or self.shoves or target is None:
            return  # the human may interact only once per episode
        origin = self.env.pos
        dest, reward, _ = self.env.shove(*target)
        self.total += reward
        self.agent.observe_correction(origin, *target, dest)
        self.animating, self.hover = True, None
        self.draw()
        self.animate(self.shove_frames(origin, dest), lambda: self.finish_shove(origin, dest))

    def finish_shove(self, origin: Pos, dest: Pos) -> None:
        self.animating = False
        self.shoves.append((origin, dest))
        self.trail.append(dest)
        self.canvas.delete('anim')
        self.draw()
        self.snapshot('shoved')

    # --- scripted demo -----------------------------------------------------

    def scripted_mode(self) -> None:
        # demo and replay drive the robot themselves, so turn off the interactive controls
        self.interactive = False
        for key in ('<space>', 'h', 'r', 'a'):
            self.root.unbind(key)
        self.canvas.unbind('<Button-1>')

    def run_demo(self, snapshot_dir: Optional[Path] = None, loop: bool = True) -> None:
        self.scripted_mode()
        self.loop, self.snapshot_dir, self.on_episode_end = loop, snapshot_dir, self.end_scenario
        for name in ('q-learning', 'memory'):  # train up front so the demo does not stall
            self.agent_name = name
            _ = self.agent
        self.demo_index = 0
        self.start_scenario()
        self.run()

    def start_scenario(self) -> None:
        name, hurry, with_human, caption = DEMO[self.demo_index]
        self.agent_name = name
        self.reset(hurry)
        self.human = Human(self.env, random.Random()) if with_human else None
        if self.human:
            self.human.reset()
        self.caption = f'{self.demo_index + 1}/{len(DEMO)}  {caption}'
        self.label = f'{self.demo_index + 1}-{self.agent_name}'
        self.draw()
        self.snapshot('start')
        self.root.after(DEMO_PAUSE_MS, self.demo_tick)

    def end_scenario(self) -> None:
        self.demo_index += 1
        if self.demo_index == len(DEMO):
            if not self.loop:
                self.root.destroy()
                return
            self.demo_index = 0
        self.root.after(DEMO_PAUSE_MS, self.start_scenario)

    # --- replaying experiment runs -------------------------------------------

    def run_replay(self, selectors: list[str], once: bool = False, snapshot_dir: Optional[Path] = None) -> None:
        # selectors are run IDs (m6, q3), an agent prefix (m, q) for all of its runs, or 'all'
        self.scripted_mode()
        self.once, self.snapshot_dir, self.on_episode_end = once, snapshot_dir, self.end_replay
        self.help = REPLAY_HELP
        self.replay_runs = self.select_runs(selectors)
        self.replay_index, self.mismatches = 0, []
        self.root.bind('n', lambda _: self.goto_run(self.replay_index + 1))
        self.root.bind('p', lambda _: self.goto_run(self.replay_index - 1))
        self.root.bind('<Return>', lambda _: self.goto_run(self.replay_index))
        self.start_replay()
        self.run()

    def select_runs(self, selectors: list[str]) -> list[Run]:
        wanted = {name for name, p in PREFIX.items() for sel in selectors if sel == 'all' or sel.startswith(p)}
        runs: dict[str, Run] = {}
        for name in sorted(wanted):
            self.agent_name = name
            runs.update((run.id, run) for run in plan_runs(self.env, self.agent))
        chosen = []
        for sel in selectors:
            if sel in runs:
                chosen.append(runs[sel])
            elif sel == 'all' or sel in PREFIX.values():
                chosen += [r for r in runs.values() if sel == 'all' or r.id[0] == sel]
            else:
                raise SystemExit(f'unknown run {sel!r}; known runs: {" ".join(runs)}')
        return chosen

    def goto_run(self, index: int) -> None:
        if self.animating or not 0 <= index < len(self.replay_runs):
            return
        if getattr(self, 'pending', None):
            self.root.after_cancel(self.pending)  # stop the episode being played
        self.replay_index = index
        self.start_replay()

    def start_replay(self) -> None:
        run = self.replay_runs[self.replay_index]
        self.agent_name = run.agent
        self.expected = experiment.replay(self.env, self.agent, run)['return']  # what experiment.py reports
        self.reset(run.hurry)
        self.human = run.human(self.env)
        if self.human:
            self.human.reset()
        self.caption = (f'[{self.replay_index + 1}/{len(self.replay_runs)}]  {run.describe()}.  '
                        f'Experiment return: {self.expected:.0f}')
        self.label = run.id
        self.draw()
        self.snapshot('start')
        self.pending = self.root.after(DEMO_PAUSE_MS, self.demo_tick)

    def end_replay(self) -> None:
        run = self.replay_runs[self.replay_index]
        same = self.total == self.expected
        print(f'{run.id}: replayed return {self.total:.0f}, experiment {self.expected:.0f}'
              f'{"" if same else "  MISMATCH"}', flush=True)
        if not same:
            self.mismatches.append(run.id)
        self.caption += f'   Replayed: {self.total:.0f}' + ('' if same else '  (MISMATCH)')
        self.draw()
        if self.once:
            if self.replay_index + 1 == len(self.replay_runs):
                self.root.destroy()
            else:
                self.pending = self.root.after(DEMO_PAUSE_MS, self.goto_run, self.replay_index + 1)

    def demo_tick(self) -> None:
        self.pending = None
        if self.animating:
            self.pending = self.root.after(DEMO_STEP_MS, self.demo_tick)
            return
        if self.env.done:
            self.snapshot('end')
            self.on_episode_end()
            return
        shove = self.human.intervene(self.agent.act(self.env.observe())) if self.human else None
        if shove:
            (dr, dc), dist = MOVES[shove[0]], shove[1]
            self.shove_to((self.env.pos[0] + dr * dist, self.env.pos[1] + dc * dist))
        else:
            self.policy_step()
        self.pending = self.root.after(DEMO_STEP_MS, self.demo_tick)

    def snapshot(self, moment: str) -> None:
        if self.snapshot_dir is None:
            return
        self.root.update_idletasks()  # redraw only: a full update() would run pending timers mid-snapshot
        stem = self.snapshot_dir / f'{self.label}-{moment}'
        self.canvas.postscript(file=f'{stem}.ps', colormode='color')
        subprocess.run(['magick', '-density', '144', f'{stem}.ps', '-flatten', f'{stem}.png'], check=True)
        Path(f'{stem}.ps').unlink()

    def set_hover(self, cell: Optional[Pos]) -> None:
        if cell != self.hover and not self.animating:
            self.hover = cell
            self.draw()

    # --- animation ---------------------------------------------------------

    def shove_frames(self, origin: Pos, dest: Pos) -> Iterator[Frame]:
        # a hand comes down, lifts the robot, carries it over, sets it down and leaves
        (ax, ay), (bx, by) = self.center(origin), self.center(dest)
        offscreen = -40.0

        def grip(y: float, lift: float) -> float:
            return y - lift - 18  # palm sits just above the robot's head

        for t in ease(12):
            yield ax, ay, 0, lerp(offscreen, grip(ay, 0), t)
        for t in ease(8):
            yield ax, ay, LIFT * t, grip(ay, LIFT * t)
        for t in ease(8 + 6 * round((abs(bx - ax) + abs(by - ay)) / CELL)):
            x, y = lerp(ax, bx, t), lerp(ay, by, t)
            yield x, y, LIFT, grip(y, LIFT)
        for t in ease(8):
            yield bx, by, LIFT * (1 - t), grip(by, LIFT * (1 - t))
        for t in ease(12):
            yield bx, by, 0, lerp(grip(by, 0), offscreen, t)

    def animate(self, frames: Iterator[Frame], on_done: Callable[[], None]) -> None:
        frame = next(frames, None)
        if frame is None:
            on_done()
            return
        x, y, lift, hand_y = frame
        self.canvas.delete('anim')
        self.draw_robot(x, y, lift, tags='anim')
        self.draw_hand(x, hand_y, tags='anim')
        self.root.after(FRAME_MS, self.animate, frames, on_done)

    # --- geometry ----------------------------------------------------------

    @staticmethod
    def cell_at(event: tk.Event) -> Pos:
        return event.y // CELL, event.x // CELL

    @staticmethod
    def box(pos: Pos, pad: int = 0) -> tuple[int, int, int, int]:
        r, c = pos
        return c * CELL + pad, r * CELL + pad, (c + 1) * CELL - pad, (r + 1) * CELL - pad

    @staticmethod
    def center(pos: Pos) -> tuple[float, float]:
        return (pos[1] + 0.5) * CELL, (pos[0] + 0.5) * CELL

    def round_rect(self, x0: float, y0: float, x1: float, y1: float, r: float, **kw) -> None:
        pts = (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
               x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0)
        self.canvas.create_polygon(pts, smooth=True, **kw)

    def diamond(self, x: float, y: float, w: float, h: float, **kw) -> None:
        self.canvas.create_polygon(x, y - h, x + w, y, x, y + h, x - w, y, **kw)

    # --- static scene ------------------------------------------------------

    def draw_room(self) -> None:
        env, cv = self.env, self.canvas
        for r in range(env.size):
            for c in range(env.size):
                cv.create_rectangle(*self.box((r, c)), fill=COLORS['floor'], outline=COLORS['grid'])
        if env.rug:
            self.draw_rug()
        self.draw_dock(*self.center(env.start))
        self.draw_flag(*self.center(env.goal))

    def draw_rug(self) -> None:
        cv, rows, cols = self.canvas, [r for r, _ in self.env.rug], [c for _, c in self.env.rug]
        x0, y0, _, _ = self.box((min(rows), min(cols)))
        _, _, x1, y1 = self.box((max(rows), max(cols)))
        border, dark, accent = COLORS['rug_border'], COLORS['rug_dark'], COLORS['rug_accent']

        # fringe on the two short ends
        if y1 - y0 >= x1 - x0:
            fringe = [(x, y, x, y + d) for x in range(x0 + 6, x1 - 4, 5) for y, d in ((y0 + 2, -6), (y1 - 2, 6))]
        else:
            fringe = [(x, y, x + d, y) for y in range(y0 + 6, y1 - 4, 5) for x, d in ((x0 + 2, -6), (x1 - 2, 6))]
        for line in fringe:
            cv.create_line(*line, fill=border, width=2)

        cv.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, fill=COLORS['rug'], outline=dark, width=2)
        cv.create_rectangle(x0 + 9, y0 + 9, x1 - 9, y1 - 9, outline=border, width=3)
        cv.create_rectangle(x0 + 15, y0 + 15, x1 - 15, y1 - 15, outline=dark)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.3, (y1 - y0) * 0.3
        for k, fill in ((1, border), (0.65, accent), (0.3, border), (0.1, dark)):
            self.diamond(cx, cy, w * k, h * k, fill=fill)
        for mx, my in ((x0 + 30, y0 + 30), (x1 - 30, y0 + 30), (x0 + 30, y1 - 30), (x1 - 30, y1 - 30)):
            self.diamond(mx, my, 8, 8, fill=accent, outline=border)

    def draw_dock(self, x: float, y: float) -> None:
        # charging dock at the start
        self.round_rect(x - 12, y - 12, x + 12, y + 12, 6, fill=COLORS['dock'], outline=COLORS['robot_dark'])
        bolt = (3, -10, -6, 2, -1, 2, -3, 10, 6, -2, 1, -2)
        self.canvas.create_polygon([v + (x if i % 2 == 0 else y) for i, v in enumerate(bolt)], fill=COLORS['light'])

    def draw_flag(self, x: float, y: float) -> None:
        # checkered flag at the goal
        cv, dark, sq = self.canvas, COLORS['robot_dark'], 4.5
        cv.create_oval(x - 9, y + 11, x - 3, y + 15, fill=dark, outline='')
        cv.create_line(x - 6, y + 13, x - 6, y - 14, fill=dark, width=2)
        fx, fy = x - 5, y - 14
        for i in range(4):
            for j in range(3):
                cv.create_rectangle(fx + i * sq, fy + j * sq, fx + (i + 1) * sq, fy + (j + 1) * sq, outline='',
                                    fill=dark if (i + j) % 2 else 'white')
        cv.create_rectangle(fx, fy, fx + 4 * sq, fy + 3 * sq, outline=dark)

    # --- dynamic scene -----------------------------------------------------

    def draw_robot(self, x: float, y: float, lift: float = 0, tags: str = 'dynamic') -> None:
        cv, dark, lifted = self.canvas, COLORS['robot_dark'], lift > 0
        k = 1 - lift / (3 * LIFT)  # shadow shrinks as the robot rises
        cv.create_oval(x - 11 * k, y + 14, x + 11 * k, y + 18, fill=COLORS['shadow'], outline='', tags=tags)
        y -= lift

        for wx in (x - 7, x + 7):  # wheels
            cv.create_oval(wx - 4, y + 10, wx + 4, y + 17, fill=dark, outline='', tags=tags)
        arm_dy = -6 if lifted else 4  # arms go up when picked up
        for side in (-1, 1):
            cv.create_line(x + 10 * side, y + 6, x + 15 * side, y + 6 + arm_dy, fill=dark, width=2,
                           capstyle='round', tags=tags)
        body = COLORS['robot_hurry'] if self.env.hurry else COLORS['robot']
        self.round_rect(x - 11, y + 1, x + 11, y + 13, 4, fill=body, outline=dark, tags=tags)
        cv.create_oval(x - 2.5, y + 4.5, x + 2.5, y + 9.5, fill=COLORS['light'], outline=dark, tags=tags)

        cv.create_line(x, y - 9, x, y - 14, fill=dark, width=2, tags=tags)  # antenna
        cv.create_oval(x - 2.5, y - 17, x + 2.5, y - 12, fill='#e74c3c', outline='', tags=tags)
        self.round_rect(x - 9, y - 10, x + 9, y + 2, 4, fill=COLORS['robot_head'], outline=dark, tags=tags)
        er = 2.6 if lifted else 1.8  # wide eyes when surprised
        for ex in (x - 4, x + 4):
            cv.create_oval(ex - er, y - 6 - er, ex + er, y - 6 + er, fill=dark, outline='', tags=tags)
        if lifted:
            cv.create_oval(x - 1.5, y - 3, x + 1.5, y, outline=dark, tags=tags)
        else:
            cv.create_arc(x - 3, y - 5, x + 3, y, start=200, extent=140, style='arc', outline=dark, tags=tags)

    def draw_hand(self, x: float, y: float, tags: str) -> None:
        # an arm reaching down from above, palm centred at (x, y), fingers curling over the robot's head
        cv, skin = self.canvas, COLORS['skin']
        cv.create_line(x, -60, x, y - 12, fill=COLORS['sleeve'], width=13, tags=tags)
        cv.create_line(x - 7, y - 12, x + 7, y - 12, fill='white', width=3, tags=tags)
        cv.create_line(x, y - 11, x, y - 3, fill=skin, width=9, tags=tags)
        cv.create_line(x - 9, y - 1, x - 12, y + 6, fill=skin, width=4, capstyle='round', tags=tags)  # thumb
        cv.create_oval(x - 9, y - 6, x + 9, y + 5, fill=skin, outline=COLORS['skin_dark'], tags=tags)
        for fx in (x - 6, x - 2, x + 2, x + 6):
            cv.create_line(fx, y + 1, fx, y + 9, fill=skin, width=4, capstyle='round', tags=tags)

    def draw(self) -> None:
        cv = self.canvas
        cv.delete('dynamic')

        routes = self.agent.candidate_routes()
        for route in routes[:MAX_ROUTES_DRAWN]:  # the route of the group of experience being followed
            cv.create_line(*[v for s in route.states for v in self.center(s)], fill=COLORS['memory'],
                           width=2, tags='dynamic')

        if self.interactive and not (self.env.done or self.animating or self.shoves):
            targets = self.shove_targets()
            for cell in targets:
                cv.create_rectangle(*self.box(cell, pad=2), outline=COLORS['shove'],
                                    width=3 if cell == self.hover else 1, tags='dynamic')
            if self.hover in targets:
                cv.create_line(*self.center(self.env.pos), *self.center(self.hover), fill=COLORS['shove'],
                               arrow='last', tags='dynamic')

        for pos in self.trail[1:]:  # skip the start so the dock stays visible
            cv.create_oval(*self.box(pos, pad=13), fill=COLORS['trail'], outline='', tags='dynamic')
        for origin, dest in self.shoves:
            cv.create_line(*self.center(origin), *self.center(dest), fill=COLORS['shove'], width=3, arrow='last',
                           tags='dynamic')
        if not self.animating:
            self.draw_robot(*self.center(self.env.pos))
        if self.caption:
            width = self.env.size * CELL
            cv.create_rectangle(0, 0, width, 2 * CELL, fill=COLORS['banner'], outline='', tags='dynamic')
            cv.create_text(10, CELL, text=self.caption, anchor='w', width=width - 20, font=('Helvetica', 13),
                           fill=COLORS['robot_dark'], tags='dynamic')
        cv.tag_raise('anim')

        state = 'done' if self.env.done else ''
        self.status['text'] = (
            f'agent: {self.agent_name:<10}  hurry: {self.env.hurry!s:<5}  steps: {self.env.steps:<3}  '
            f'return: {self.total:<6.0f}  shove used: {bool(self.shoves)!s:<5}  following group: {getattr(self.agent, "chosen", None)!s:<4}'
            f'{state}\n{self.help}'
        )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == '__main__':
    if tk.TkVersion < 8.6:
        print(f'Warning: Tk {tk.TkVersion} is outdated and may show a blank window on recent macOS; '
              'use a Python with Tk >= 8.6 (e.g. one installed via uv or python.org).')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='play the scripted scenarios on a loop')
    parser.add_argument('--replay', nargs='+', metavar='RUN',
                        help='replay experiment.py runs by ID (e.g. m6 q3), all runs of an agent (m, q), or all')
    parser.add_argument('--once', action='store_true',
                        help='with --demo or --replay: play each scenario once, then exit')
    parser.add_argument('--snapshots', type=Path,
                        help='with --demo or --replay: save a PNG at key moments to this folder')
    args = parser.parse_args()
    random.seed(0)
    gui = LivingRoomGUI(LivingRoom())
    if args.snapshots:
        args.snapshots.mkdir(parents=True, exist_ok=True)
    if args.replay:
        gui.run_replay(args.replay, once=args.once, snapshot_dir=args.snapshots)
        if gui.mismatches:
            raise SystemExit(f'replays differ from experiment.py: {" ".join(gui.mismatches)}')
    elif args.demo:
        gui.run_demo(args.snapshots, loop=not args.once)
    else:
        gui.run()
