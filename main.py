# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 07:58:43 2025

Ziply is a directional graph search problem where every node must be reached 
without intersecting visited nodes and requiring checkpoints in sequence.

This code is intended to work cross-platform, however, linux users must be using
x11 for pyautogui and mss, and otherwise is untested in that environment.

At the beginning of this project, I used camel case for variables and 
snake case for functions. I have since switched to using all snake_case
when coding in Python. The switched conventions can still be seen, especially in
game_data.py, which was the initial module of this program.

@author: barna
"""
# ----- Import Start ----- #
# Silence info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

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

# Local Modules
from utils.dependents.compat import check_environment
from visualization_utils.board_anim import live_animation
from visualization_utils.intersection_heatmap import intersection_heatmap
from game_data import GameData
from utils.cli_parser import parse_arguments
from solvers.forwards_dfs import solve_puzzle

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
        
def create_gameboard(H, W):
    board = np.zeros((H, W), dtype=int)
    
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
            add_pt(path[point_index])   # pushes the point into the deque
            point_index += 1
        else:
            # stop time after all points sent
            timer_obj.stop()
        
    return feed_next_point

def main():
    args = parse_arguments()
    
    displayHeatmap = args.displayHeatmap
    displayAnimation = args.displayAnimation
    drawSolution = args.drawSolution
    displaySolutionCoords = args.displaySolutionCoords
    ts = args.ts
    simulationLength = args.simulationLength  # Will be 1000 by default
    
    print(f"Simulation will run for {simulationLength} coordinates.")
    
    # ----- argparser end ----- #
    
    check_environment()
    moves = None
    visited = None
    print("Click the window containing the puzzle... ")
    wait_for_click()
    time.sleep(1)
    tk_board = False
    H = 6
    W = 6
    
    try:
        data = (
            GameData(H, W, ts, tk_board = tk_board) # data pipeline
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
        board = create_gameboard(H, W)
        #print(f"final grid locations: {data.grid_locations}")
        
        board, displayBoard = populate_gameboard(data.grid_locations, board)
        #print(data.grid_locations)
        
        # Time solving the path
        startTime = time.time()
        solution = solve_puzzle(
            board,
            simulationLength = simulationLength
            )
        
        endTime = time.time()
        elapsedTime = endTime - startTime
        
        # Good if extracting a game board
        if displaySolutionCoords:
            showBoard = np.array2string(displayBoard, separator = ', ')
            print(f"Game Board:\n{showBoard}\n")
            print(f"Circle Coordinates: {data.grid_locations}")
            
            print(f"Solution path:\n{solution}\n")
            
        print(f"Time to solve: {elapsedTime:.3f}s")
        
        if moves is not None:
            print(f"Total moves: {moves}")
        
        if solution is not None:
            if isinstance(solution, tuple):
                data.grid_to_pixels(solution[0]).get_absolute_coords()
            else:
                data.grid_to_pixels(solution).get_absolute_coords()
            
            # Run puzzle solving and solution drawing at the same time
            puzzle_thread = threading.Thread(
                target = complete_puzzle,
                args=(data.pixel_coords,)
                )
            
            if drawSolution:
                puzzle_thread.start()
            
            if displayHeatmap:
                intersection_heatmap(displayBoard, data.grid_locations)
                
                plt.show()
            
            if displayAnimation:
                # Create animation
                anim, add_point = live_animation(
                        displayBoard,
                        interval=50,
                        line_width=2,
                        line_color='crimson',
                        line_alpha=0.5,
                        draw_arrows=True,
                        fps=12,
                        fade_frames = 100)
                
                if visited is not None:
                    # Fire timer ever 50ms
                    timer = anim._fig.canvas.new_timer(interval=50)
                    timer.add_callback(make_feeder(visited, add_point, timer))
                    timer.start()
    
                    plt.show()
                
        else:
            raise Exception("Solver returned None!\nLines already drawn - refresh the puzzle.\nIf you keep getting this error, resize the window.")
            
            
    except KeyboardInterrupt:
        print("Exiting... ")
        

    
if __name__ == "__main__":
    main()