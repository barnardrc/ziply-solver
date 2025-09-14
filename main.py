# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 07:58:43 2025

Ziply is a directional graph search problem where every node must be reached 
without intersecting visited nodes and requiring checkpoints in sequence.

This code is intended to work cross-platform, however, linux users must be using
x11 for pyautogui and mss, and otherwise is untested in that environment.

@author: barna
"""
# ----- Import Start ----- #

# Force mpl backend
import matplotlib
matplotlib.use('TkAgg')

# Packages
import threading
import time
from pynput import mouse
import numpy as np
import pyautogui as pag
import matplotlib.pyplot as plt
import argparse

# Local Modules
from compat import check_environment
from animation_utils.board_anim import live_animation
from game_data import GameData
from solvers import backtrack_dfs

# ----- Import End ----- #

def wait_for_click():
    
    click_event = threading.Event()
    
    def _on_click(x, y, button, pressed):
    
        if pressed:
            click_event.set()
            return False
    
    with mouse.Listener(on_click=_on_click) as listener:
        listener.join()
    
    click_event.wait()
        
def create_gameboard():
    board = np.zeros((6, 6), dtype=int)
    
    return board

def populate_gameboard(list_of_coords, board):
    #print(f"in populate_gameboard: {list_of_coords}")
    displayBoard = board.copy()
    for i, (x, y) in enumerate(list_of_coords):
        board[x, y] = i+1
        displayBoard[y, x] = i+1
    
    return board, displayBoard

def coords_to_directions(path, moveset):
    directions = []
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        directions.append(moveset[(dx, dy)])
    return directions

def complete_puzzle(absolute_coords):
    pag.moveTo(absolute_coords[0])
    pag.mouseDown()
    for (x, y) in absolute_coords:
        pag.moveTo(x, y)
    pag.mouseUp()

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

def main():
    # ----- argparser ----- #
    
    parser = argparse.ArgumentParser(description="A solver for the Ziply directional graph puzzle.")
    
    # Display animation that simulates the path the algorithm takes
    parser.add_argument(
        '-na','--no-animation', 
        action='store_false', 
        dest='displayAnimation',
        help="Disable the final path animation."
    )
    
    # Draw solution in browser
    parser.add_argument(
        '-ns','--no-solution', 
        action='store_false',
        dest='drawSolution',
        help="Disable drawing in the puzzle window."
    )
    
    # Print solution coords to console
    parser.add_argument(
        '-dc','--display-coords', 
        action='store_true',
        dest='displaySolutionCoords',
        help="Print the final solution coordinates to the console."
    )
    # Change simulation length
    parser.add_argument(
        '-sl', '--sim-length',
        type=int,
        default=100,
        dest='simulationLength',
        help = "First simulationLength coordinates will be simulated: sl={int}")
    
    args = parser.parse_args()
    
    displayAnimation = args.displayAnimation
    drawSolution = args.drawSolution
    displaySolutionCoords = args.displaySolutionCoords
    simulationLength = args.simulationLength  # Will be 100 by default
    print(f"Simulation will run for {simulationLength} coordinates.")
    
    # ----- argparser end ----- #
    
    check_environment()
    
    # Move set
    moves = {
        (-1, 0): "LEFT",
        (1, 0): "RIGHT",
        (0, -1): "UP",
        (0, 1): "DOWN"
    }
    
    print("Click the window containing the puzzle... ")
    wait_for_click()
    time.sleep(1)
    

    try:
        data = (
            GameData() # data pipeline
                 #.toggle_ts_mode() #ts_mode largely provides a step by step
                                     # of what is happening
                 .get_window_rect()
                 .window_capture()
                 .detect_circles()
                 .detect_clusters()
                 .get_circle_region_bounds()
                 .get_roi()
                 .canny_edge()
                 .set_board_edges()
                 .fill_background()
                 .get_squares()
                 .extract_square_images()
                 .predict_digits()
                 .order_circles()
                 .pixels_to_grid()
         )
        
        # Create the board and populate it with checkpoints
        board = create_gameboard()
        #print(f"final grid locations: {data.grid_locations}")
        
        board, displayBoard = populate_gameboard(data.grid_locations, board)

        
        # Time solving the path
        startTime = time.time()
        solution, recursions, visited = backtrack_dfs(
            board,
            data.grid_locations, 
            moves,
            simulationLength
            )
        
        endTime = time.time()
        elapsedTime = endTime - startTime
        
        if displaySolutionCoords:
            print(f"Game Board:\n{displayBoard}\n")
            
            print(f"Solution path:\n{solution}\n")
            
        print(f"Time to solve: {elapsedTime:.3f}s")
        print(f"Total recursions: {recursions}")
        
        if solution is not None:
            data.grid_to_pixels(solution).get_absolute_coords()
            
            # Run puzzle solving and solution drawing at the same time
            puzzle_thread = threading.Thread(
                target = complete_puzzle,
                args=(data.pixel_coords,)
                )
            
            if drawSolution:
                puzzle_thread.start()
                
            if displayAnimation:
                # Create animation
                anim, add_point = live_animation(
                        displayBoard,
                        interval=50,               # ms between frames → ~25 fps
                        line_width=2,
                        line_color='crimson',
                        line_alpha=0.5,
                        draw_arrows=True,
                        fps=12,
                        fade_frames = 100)
                
                # Fire timer ever 50ms
                timer = anim._fig.canvas.new_timer(interval=50)
                timer.add_callback(make_feeder(visited, add_point, timer))
                timer.start()

                plt.show()
            
        else:
            raise Exception("Lines already drawn - refresh the puzzle.\n If you keep getting this error, resize the window.")
            
            
    except KeyboardInterrupt:
        print("Exiting... ")
            

    
if __name__ == "__main__":
    main()