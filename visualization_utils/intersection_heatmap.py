# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:18:52 2025

@author: barna
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle

def get_position(p1, p2):
    if p2[0] < p1[0]:
        pass
    
    
def get_rect(ordered_circles):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'steelblue', 'gold']
    rectangles = []
    alpha = 0.3
    
    for i, circle in enumerate(ordered_circles):
        # Check if there is a next circle to connect to
        if i < len(ordered_circles) - 1:
            next_circle = ordered_circles[i + 1]
            
            x1_coord = circle[0]
            y1_coord = circle[1]
            x2_coord = next_circle[0]
            y2_coord = next_circle[1]

            # Determine the bottom-left and top-right corners
            x_min = min(x1_coord, x2_coord) - 0.5
            y_min = min(y1_coord, y2_coord) - 0.5
            x_max = max(x1_coord, x2_coord) + 0.5
            y_max = max(y1_coord, y2_coord) + 0.5

            width = x_max - x_min
            height = y_max - y_min

            rect = Rectangle((x_min, y_min), width, height,
                             edgecolor=colors[2],
                             facecolor=colors[3],
                             alpha=alpha)
            rectangles.append(rect)
            
            alpha += 0.04
            
    return rectangles
    
    
    

def draw_board(ax: plt.Axes,
               board: np.ndarray,
               ordered_circles: list,
               circle_facecolor: str = "k",
               circle_edgecolor: str = "purple",
               circle_linewidth: float = 3,
               circle_radius: float = 0.35) -> None:

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
    
    rectangles = get_rect(ordered_circles)
    for rectangle in rectangles:
        ax.add_patch(rectangle)
    

def intersection_heatmap(board, ordered_circles):
    
    N = board.shape[0]
    x = N-1 
    y = N-1
    # Create
    fig, ax = plt.subplots(figsize=(x, y))
    draw_board(ax, board, ordered_circles)
