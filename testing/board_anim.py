# ------------------------------------------------------------
# board_anim.py
# ------------------------------------------------------------
import itertools
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.interpolate import make_interp_spline


def draw_board(ax: plt.Axes,
               board: np.ndarray,
               circle_facecolor: str = "k",
               circle_edgecolor: str = "purple",
               circle_linewidth: float = 3,
               circle_radius: float = 0.35) -> None:
    """Same draw_board you already have – unchanged."""
    H, W = board.shape
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for i in range(H):
        ax.plot([-0.5, W - 0.5], [i - 0.5, i - 0.5],
                color='grey', linewidth=1, alpha=0.4)
    for j in range(W):
        ax.plot([j - 0.5, j - 0.5], [-0.5, H - 0.5],
                color='grey', linewidth=1, alpha=0.4)

    for r in range(H):
        for c in range(W):
            if board[r, c] != 0:
                circ = Circle((c, r), radius=circle_radius,
                              facecolor=circle_facecolor,
                              edgecolor=circle_edgecolor,
                              linewidth=circle_linewidth,
                              zorder=2)
                ax.add_patch(circ)
                ax.text(c, r, str(board[r, c]),
                        ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='white', zorder=3)


def live_animation(board: np.ndarray,
                   interval: int = 40,
                   line_width: int = 3,
                   line_color: str = "crimson",
                   line_alpha: float = 0.4,
                   draw_arrows: bool = True,
                   fps: int = 30) -> tuple[FuncAnimation, callable]:
    """
    Returns two objects:
        * the FuncAnimation instance (must stay alive)
        * a *callback* `add_point((row, col))` that you can call from any thread
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_board(ax, board)

    # -----------------------------------------------------------------
    #  shared buffer that the solver will fill
    # -----------------------------------------------------------------
    live_path = deque()
    rc2xy = lambda rc: (rc[1], rc[0])           # (row, col) → (x, y)

    # -----------------------------------------------------------------
    #  line / arrow artists
    # -----------------------------------------------------------------
    line, = ax.plot([], [], color=line_color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=5)               # keep it simple, no set_animated()

    arrow_art = []                               # at most one FancyArrowPatch

    # -----------------------------------------------------------------
    #  spline helper
    # -----------------------------------------------------------------
    def recompute_spline():
        if len(live_path) < 2:
            return np.array([]), np.array([])

        pts = np.array([rc2xy(p) for p in live_path])
        t = np.arange(len(pts))
        k = 3 if len(pts) >= 4 else 1
        spline_x = make_interp_spline(t, pts[:, 0], k=k)
        spline_y = make_interp_spline(t, pts[:, 1], k=k)

        n_dense = min(800, len(pts) * 10)
        t_dense = np.linspace(0, len(pts) - 1, n_dense)
        return spline_x(t_dense), spline_y(t_dense)

    # -----------------------------------------------------------------
    #  animation callbacks
    # -----------------------------------------------------------------
    def init():
        line.set_data([], [])
        return line,

    def update(_frame):
        xs, ys = recompute_spline()
        line.set_data(xs, ys)

        # arrow head
        for a in arrow_art:
            a.remove()
        arrow_art.clear()
        if draw_arrows and len(xs) >= 2:
            arrow = FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]),
                                    arrowstyle='->',
                                    color=line_color,
                                    mutation_scale=12,
                                    linewidth=line_width,
                                    zorder=6)
            ax.add_patch(arrow)
            arrow_art.append(arrow)

        return (line,) + tuple(arrow_art)

    anim = FuncAnimation(fig,
                        update,
                        frames=itertools.count(),
                        init_func=init,
                        interval=interval,
                        blit=False,
                        cache_frame_data=False,
                        repeat=False)

    # -----------------------------------------------------------------
    #  public helper that the solver will call
    # -----------------------------------------------------------------
    def add_point(rc):
        """Thread‑safe (deque.append is atomic in CPython)."""
        live_path.append(rc)

    return anim, add_point
