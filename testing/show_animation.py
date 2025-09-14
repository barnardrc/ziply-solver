# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:15:57 2025

@author: barna
"""

import numpy as np

# --------------------------------------------------------------
# board_anim.py
# --------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import tee
from typing import List, Tuple, Optional
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline

def _pairwise(seq):
    """Yield overlapping (a, b) pairs from a sequence."""
    a, b = tee(seq)
    next(b, None)
    return zip(a, b)

def bezier_quad(p0, p1, n=40, curvature=0.2):
    """Return xs,ys of a quadratic Bézier between p0 and p1.
    curvature (0‑1) controls how far the control point sticks out."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    # control point = midpoint displaced orthogonal to the segment
    mid = (p0 + p1) / 2
    # direction orthogonal to the line (swap, negate one component)
    ortho = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]])
    ortho = ortho / np.linalg.norm(ortho + 1e-12)   # unit vector
    c = mid + curvature * np.linalg.norm(p1 - p0) * ortho

    t = np.linspace(0, 1, n)[:, None]   # column vector
    xs = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * c[0] + t**2 * p1[0]
    ys = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * c[1] + t**2 * p1[1]
    return xs.ravel(), ys.ravel()

# Generating grid lines. It is done this way because there is a 0.5 unit offset.
def generate_lines(N: int = 6):
    lines = []
    
    # Generate axis 0 lines
    for i in range(0, N, 1):
        line = []
        
        x, y = -0.5, i + 0.5
        line.append(x), line.append(y)
        
        x, y = N + .5, i + 0.5
        line.append(x), line.append(y)
        
        lines.append(line)
    
    # Generate axis 1 lines
    for i in range(0, N, 1):
        line = []
        
        y, x = -0.5, i + 0.5
        line.append(x), line.append(y)
        
        y, x = N + .5, i + 0.5
        line.append(x), line.append(y)
        
        lines.append(line)
        
        
    print(lines)
    return lines

generate_lines()

def draw_board(ax: plt.Axes,
               board: np.ndarray,
               show_numbers: bool = True,
               circle_facecolor: str = "k",   # leave the inside transparent
               circle_edgecolor: str = "purple",    # colour of the circle outline
               circle_linewidth: float = 3,
               circle_radius: float = 0.35,     # 0.5 would touch the grid lines
               cmap: str = "Blues_r") -> None:
    """
    Render `board` and draw a **circle in every non‑zero cell**.

    Parameters
    ----------
    ax, board, show_numbers
        Same meaning as in your original implementation.
    circle_facecolor, circle_edgecolor, circle_linewidth, circle_radius
        Styling for the circles that appear around the numbers.
    cmap
        Kept only for signature compatibility; it is not used for the circles.
    """
    
    # draw a line from cell (r1,c1) to cell (r2,c2)
    def draw_line(ax, r1, c1, r2, c2, **kw):
        """r = row index (y), c = column index (x)."""
        ax.plot([c1, c2],            # x‑coordinates (columns)
                [r1, r2],            # y‑coordinates (rows)
                **kw)                # e.g. color='r', linewidth=2
    
    # examples
    #draw_line(ax, 0, 0, 5, 5, color='red',   linewidth=3)   # diagonal
    #draw_line(ax, 2, 1, 2, 4, color='green', linewidth=2)   # horizontal
    #draw_line(ax, 1, 3, 4, 3, color='blue',  linewidth=2)   # vertical
    
    
    
    H, W = board.shape

    # -----------------------------------------------------------------
    # 1️⃣  Grid & tick‑label setup (unchanged from your version)
    # -----------------------------------------------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Draw the grid lines
    lines = generate_lines()
    for line in lines:
        r1, c1, r2, c2 = line
        draw_line(ax, r1, c1, r2, c2, color = 'grey', linewidth = 2, alpha = 0.3)
    
    # each cell = 1 × 1 data unit → we can address them with integer coords
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)

    # -----------------------------------------------------------------
    # 2️⃣  Draw a circle **only** where board != 0
    # -----------------------------------------------------------------
    for r in range(H):
        for c in range(W):
            if board[r, c] != 0:                     # ← non‑zero → draw circle
                circ = Circle((c, r),                # (x, y) = (col, row)
                               radius=circle_radius,
                               facecolor=circle_facecolor,
                               edgecolor=circle_edgecolor,
                               linewidth=circle_linewidth,
                               zorder=2)           # on top of the grid
                ax.add_patch(circ)

    # -----------------------------------------------------------------
    # 3️⃣  (Optional) write the numbers inside the circles
    # -----------------------------------------------------------------
    if show_numbers:
        for (r, c), v in np.ndenumerate(board):
            if v != 0:
                ax.text(c, r, str(v),
                        ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white", zorder=3)

    # -----------------------------------------------------------------
    # 4️⃣  Final cosmetics
    # -----------------------------------------------------------------
    ax.set_aspect("equal")
    ax.invert_yaxis()

def animate_solution(board: np.ndarray,
                    path: List[Tuple[int, int]],
                    interval: int = 40,
                    line_width: int = 3,
                    line_color: str = "crimson",
                    line_alpha: float = 0.4,
                    draw_arrows: bool = True,
                    cmap: str = "Blues_r",
                    save_as: Optional[str] = None,
                    fps: int = 30) -> animation.FuncAnimation:
    """
    Animated drawing of a smooth curve that follows *path* on top of a board.
    The curve is built from a global cubic spline and revealed incrementally.
    """
    # --------------------------------------------------------------
    # 0️⃣  Figure / board background
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_board(ax, board, cmap=cmap)      # your existing board‑drawing helper

    # --------------------------------------------------------------
    # 1️⃣  Prepare the *global* cubic spline
    # --------------------------------------------------------------
    rc2xy = lambda rc: (rc[1], rc[0])            # (row, col) → (x, y)
    pts = np.array([rc2xy(p) for p in path])    # shape (N, 2)

    t_raw = np.arange(len(pts))                 # parametric index
    # cubic spline needs ≥4 points – your puzzles satisfy that
    spline_x = make_interp_spline(t_raw, pts[:, 0], k=3)
    spline_y = make_interp_spline(t_raw, pts[:, 1], k=3)

    # dense sampling of the whole curve (the “pen” will travel along this)
    N_dense = 1200                               # increase for ultra‑smooth
    t_dense = np.linspace(0, len(pts) - 1, N_dense)
    xs_dense = spline_x(t_dense)
    ys_dense = spline_y(t_dense)

    # --------------------------------------------------------------
    # 2️⃣  Artists that will be updated each frame
    # --------------------------------------------------------------
    #   – line (the curve we are drawing)
    line, = ax.plot([], [], color=line_color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=5)

    #   – optional arrowhead (created / removed each frame)
    arrow_art = []          # list with **zero or one** FancyArrowPatch

    # how many new points appear per frame → controls speed / smoothness
    points_per_frame = 12

    # --------------------------------------------------------------
    # 3️⃣  Animation callbacks
    # --------------------------------------------------------------
    def init():
        line.set_data([], [])
        return line,

    def update(frame_idx):
        # 3a️⃣  Determine how far along the dense curve we are
        last = min((frame_idx + 1) * points_per_frame, N_dense)

        # 3b️⃣  Update the line data (the "pen" trail)
        line.set_data(xs_dense[:last], ys_dense[:last])

        # 3c️⃣  (Re)draw the tiny arrowhead that points to the *current* tip
        #      – remove the previous one first
        for art in arrow_art:
            art.remove()
        arrow_art.clear()

        if draw_arrows and last >= 2:          # need at least two points
            # start point = previous point, end point = current tip
            x0, y0 = xs_dense[last - 2], ys_dense[last - 2]
            x1, y1 = xs_dense[last - 1], ys_dense[last - 1]

            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle='->',
                color=line_color,
                mutation_scale=12,
                linewidth=line_width,
                zorder=6
            )
            ax.add_patch(arrow)
            arrow_art.append(arrow)

        # return everything that changed this frame
        return (line,) + tuple(arrow_art)

    # --------------------------------------------------------------
    # 4️⃣  Build the FuncAnimation object
    # --------------------------------------------------------------
    total_frames = int(np.ceil(N_dense / points_per_frame))
    anim = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False
    )

    # --------------------------------------------------------------
    # 5️⃣  Optional saving
    # --------------------------------------------------------------
    if save_as:
        ext = save_as.split(".")[-1].lower()
        if ext == "gif":
            anim.save(save_as, writer='pillow', fps=1000 / interval)
        elif ext == "mp4":
            anim.save(save_as, writer='ffmpeg',
                      fps=fps, codec='libx264', bitrate=1800)
        elif ext == "png":
            # draw the final frame and save it
            update(total_frames - 1)
            fig.savefig(save_as, bbox_inches='tight')
        else:
            raise ValueError("Unsupported extension. Use .gif, .mp4 or .png")
        print(f"✅ Saved animation → {save_as}")

    # plt.show()      # uncomment for interactive debugging
    return anim




board = [[0, 3, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 2],
 [0, 1, 0, 0, 0, 0],
 [0, 0, 5, 0, 0, 6],
 [4, 0, 0, 0, 8, 0],
 [0, 0, 7, 0, 0, 0]]

def list_to_array(input_list):
    return np.array(input_list)

board = list_to_array(board)

path = [(2, 1), (1, 1), (1, 2), (2, 2), (2, 3), (1, 3), (1, 4), (2, 4), 
            (2, 5), (1, 5), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0),
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (4, 1), (3, 1),
            (3, 2), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3),
            (5, 2), (4, 2), (4, 3), (4, 4)]

interval = (len(path) - len(path) // 6)
anim = animate_solution(board, path, interval = interval)

plt.show()
