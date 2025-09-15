# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 12:29:59 2025

@author: barna
"""

from board_anim import live_animation
import matplotlib.pyplot as plt
import numpy as np

board = np.array([[0, 3, 0, 0, 6, 0],
 [0, 0, 0, 0, 0, 0],
 [2, 0, 4, 0, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 5, 0, 7],
 [1, 8, 0, 0, 0, 0]])


demo_path = [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (4, 2), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2), (5, 1), (5, 4), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (1, 5), (0, 5), (0, 4), (2, 5), (3, 5), (4, 5), (2, 5), (1, 5), (0, 5), (0, 4), (1, 4), (1, 4), (0, 4), (0, 5), (3, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5), (0, 4), (1, 4), (2, 4), (1, 4), (0, 4), (0, 5), (2, 4), (2, 4), (1, 4), (0, 4), (0, 5), (1, 5), (1, 5), (0, 5), (0, 4), (4, 5), (4, 5), (5, 5), (4, 5), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (4, 2), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2), (5, 1), (5, 4), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (1, 5), (0, 5), (0, 4), (2, 5), (3, 5), (4, 5), (2, 5), (1, 5), (0, 5), (0, 4), (1, 4), (1, 4), (0, 4), (0, 5), (3, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5), (0, 4), (1, 4), (2, 4), (1, 4), (0, 4), (0, 5), (2, 4), (2, 4), (1, 4), (0, 4), (0, 5), (1, 5), (1, 5), (0, 5), (0, 4), (4, 5), (4, 5), (5, 5), (4, 5), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1)]

# Create animation
anim, add_point = live_animation(
        board,
        interval=50,               # ms between frames → ~25 fps
        line_width=2,
        line_color='crimson',
        line_alpha=0.5,
        draw_arrows=True,
        fps=12,
        fade_frames = 35)

def make_feeder(path, add_pt, timer_obj):
    point_index = 0
    
    def feed_next_point():
        """Timer callback – add one point per call."""
        nonlocal point_index
        if point_index < len(path):
            add_pt(path[point_index])   # <-- pushes the point into the deque
            point_index += 1
        else:
            # all points have been sent – stop the timer
            timer_obj.stop()
        
    return feed_next_point

# Fire timer ever 50ms
timer = anim._fig.canvas.new_timer(interval=50)
timer.add_callback(make_feeder(demo_path, add_point, timer))
timer.start()

plt.show()