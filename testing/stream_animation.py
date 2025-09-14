# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 15:44:08 2025
Author: barna (adapted)
"""

import itertools
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.interpolate import make_interp_spline

# --------------------------------------------------------------
# 1️⃣  Board drawing (circles around non‑zero cells)
# --------------------------------------------------------------
def draw_board(ax: plt.Axes,
               board: np.ndarray,
               circle_facecolor: str = "k",
               circle_edgecolor: str = "purple",
               circle_linewidth: float = 3,
               circle_radius: float = 0.35,
               cmap: str = "Blues_r") -> None:
    H, W = board.shape

    # hide ticks & tick‑labels
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # draw the light grey grid (0.5‑unit offset to hit cell borders)
    for i in range(H):
        ax.plot([-0.5, W - 0.5], [i - 0.5, i - 0.5],
                color='grey', linewidth=1, alpha=0.4)
    for j in range(W):
        ax.plot([j - 0.5, j - 0.5], [-0.5, H - 0.5],
                color='grey', linewidth=1, alpha=0.4)

    # circles + numbers for every non‑zero cell
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


# --------------------------------------------------------------
# 2️⃣  Real‑time animation
# --------------------------------------------------------------
def live_animation(board: np.ndarray,
                   interval: int = 40,          # ms between refreshes
                   line_width: int = 3,
                   line_color: str = "crimson",
                   line_alpha: float = 0.4,
                   draw_arrows: bool = True,
                   fps: int = 30,
                   save_as: str | None = None) -> FuncAnimation:

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_board(ax, board)

    # -----------------------------------------------------------------
    # Buffer that will receive the incoming (row, col) points.
    # In a real program you would call `add_point((r, c))` from your
    # socket / GUI / sensor callback.
    # -----------------------------------------------------------------
    live_path = deque()               # mutable list of received points
    rc2xy = lambda rc: (rc[1], rc[0])   # (row, col) → (x, y)

    # -----------------------------------------------------------------
    # Artists that we will update on every frame
    # -----------------------------------------------------------------
    line, = ax.plot([], [], color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=5)
    line.set_animated(True)          # <<< tell Matplotlib this artist will be animated
    
    arrow_art = []                    # will contain at most one FancyArrowPatch


    # Helper that recomputes the spline from the points we have so far
    # -----------------------------------------------------------------
    def recompute_spline():
        """Return xs, ys – dense points of the current spline."""
        if len(live_path) < 2:
            return np.array([]), np.array([])
    
        pts = np.array([rc2xy(p) for p in live_path])
        t   = np.arange(len(pts))
    
        # cubic when we have ≥4 points, otherwise linear
        k = 3 if len(pts) >= 4 else 1
    
        # <-- **no `extrapolate=False` here**
        spline_x = make_interp_spline(t, pts[:, 0], k=k)
        spline_y = make_interp_spline(t, pts[:, 1], k=k)
    
        # density = 10× number of points (capped at 800 for speed)
        n_dense = min(800, len(pts) * 10)
        t_dense = np.linspace(0, len(pts) - 1, n_dense)
        return spline_x(t_dense), spline_y(t_dense)

    # -----------------------------------------------------------------
    # FuncAnimation callbacks
    # -----------------------------------------------------------------
    def init():
        line.set_data([], [])
        return line,

    def update(_frame):
        xs, ys = recompute_spline()
        line.set_data(xs, ys)

        # ---- arrow head -------------------------------------------------
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

    total_frames = 5000                # effectively “infinite”
    anim = FuncAnimation(fig, update,
                         frames=itertools.count(),
                         init_func=init,
                         interval=interval,
                         blit=True,
                         cache_frame_data=False,
                         repeat=False)

    # --------------------------------------------------------------
    # Optional saving (same as your original implementation)
    # --------------------------------------------------------------
    if save_as:
        ext = save_as.split(".")[-1].lower()
        if ext == "gif":
            anim.save(save_as, writer='pillow', fps=1000 / interval)
        elif ext == "mp4":
            anim.save(save_as, writer='ffmpeg',
                      fps=fps, codec='libx264', bitrate=1800)
        elif ext == "png":
            # draw final frame then save
            update(0)
            fig.savefig(save_as, bbox_inches="tight")
        else:
            raise ValueError("Unsupported extension. Use .gif, .mp4 or .png")
        print(f"✅ Saved → {save_as}")

    # --------------------------------------------------------------
    # 3️⃣  Helper for feeding points (simulated here with a thread)
    # --------------------------------------------------------------
    def add_point(new_rc):
        """Public function – call this whenever a new (row, col) arrives."""
        live_path.append(new_rc)

    # -----------------------------------------------------------------
    # Demo: feed points with a tiny delay to illustrate “real‑time”.
    # Replace this block with your actual data source.
    # -----------------------------------------------------------------
    demo_path = [(2, 1), (1, 1), (1, 2), (2, 2), (2, 3), (1, 3), (1, 4), (2, 4), 
                (2, 5), (1, 5), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0),
                (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (4, 1), (3, 1),
                (3, 2), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3),
                (5, 2), (4, 2), (4, 3), (4, 4)]

    def feed_demo():
        for p in demo_path:
            add_point(p)
            time.sleep(0.1)          # simulate network / sensor latency

    threading.Thread(target=feed_demo, daemon=True).start()

    return anim


# --------------------------------------------------------------
# Run everything
# --------------------------------------------------------------
board = np.array([[0, 3, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 2],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 5, 0, 0, 6],
                  [4, 0, 0, 0, 8, 0],
                  [0, 0, 7, 0, 0, 0]])

anim = live_animation(board,
               interval=40,          # faster refresh → smoother visual
               line_width=3,
               line_color='crimson',
               line_alpha=0.5,
               draw_arrows=True,
               fps=30,
               save_as=None)         # give a filename to save (gif/mp4/png)

plt.show()
