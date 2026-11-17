"""Tkinter viewer for LivingRoom: step through the optimal policy and shove the robot as a human would."""
from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Iterator, Optional

from livingroom import MOVES, Action, LivingRoom, Pos

CELL = 34  # the artwork below is sized for this cell size
FRAME_MS = 16
LIFT = 12  # px the hand lifts the robot during a shove
COLORS = {
    'floor': '#eadcc2', 'grid': '#dccbab', 'shadow': '#bfae8e', 'trail': '#7f9cc7', 'shove': '#8e44ad',
    'rug': '#a8432f', 'rug_border': '#e8c07a', 'rug_dark': '#6e2619', 'rug_accent': '#2f5d7a',
    'robot': '#4a78c2', 'robot_hurry': '#e0662f', 'robot_head': '#d9dee6', 'robot_dark': '#2b2f3a',
    'light': '#ffd166', 'dock': '#9aa5b1', 'skin': '#f2c29b', 'skin_dark': '#c98f6b', 'sleeve': '#6c5ce7',
}
HELP = 'space: policy step   click a highlighted cell: shove   h: toggle hurry   r: reset'
Frame = tuple[float, float, float, float]  # robot x, robot y, lift, hand y


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
        self.canvas.bind('<Button-1>', lambda e: self.shove_to(self.cell_at(e)))
        self.canvas.bind('<Motion>', lambda e: self.set_hover(self.cell_at(e)))
        self.canvas.bind('<Leave>', lambda _: self.set_hover(None))

        self.draw_room()
        self.reset()

    # --- interaction -------------------------------------------------------

    def reset(self, hurry: Optional[bool] = None) -> None:
        if self.animating:
            return
        self.env.reset(hurry)
        self.trail, self.shoves, self.total = [self.env.pos], [], 0.0
        self.draw()

    def policy_step(self) -> None:
        if self.animating or self.env.done:
            return
        pos, reward, _, _ = self.env.step(self.env.optimal_action())
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
        if self.animating or self.env.done or target is None:
            return
        origin = self.env.pos
        dest = self.env.shove(*target)
        self.animating, self.hover = True, None
        self.draw()
        self.animate(self.shove_frames(origin, dest), lambda: self.finish_shove(origin, dest))

    def finish_shove(self, origin: Pos, dest: Pos) -> None:
        self.animating = False
        self.shoves.append((origin, dest))
        self.trail.append(dest)
        self.canvas.delete('anim')
        self.draw()

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

        if not (self.env.done or self.animating):
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
        cv.tag_raise('anim')

        state = 'done' if self.env.done else ''
        self.status['text'] = (
            f'hurry: {self.env.hurry!s:<5}  steps: {self.env.steps:<3}  return: {self.total:<6.0f}  '
            f'shoves: {len(self.shoves):<3}  {state}\n{HELP}'
        )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == '__main__':
    if tk.TkVersion < 8.6:
        print(f'Warning: Tk {tk.TkVersion} is outdated and may show a blank window on recent macOS; '
              'use a Python with Tk >= 8.6 (e.g. one installed via uv or python.org).')
    LivingRoomGUI(LivingRoom()).run()
