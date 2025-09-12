# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 07:58:43 2025

Ziply is a directional graph search problem where every node must be reached 
without intersecting visited nodes and requiring checkpoints in sequence.

This code is intended to work cross-platform, however, linux users must be using
x11 for pyautogui and mss, and otherwise is untested in that environment.

@author: barna
"""

import time
from pynput import mouse
from game_data import GameData
from compat import check_environment
import numpy as np
import pyautogui as pag

def on_click(x, y, button, pressed):
    global click_received
    #print(f"Position Clicked: {x}, {y}.")
    if pressed:
        click_received = True
        return False

def wait_for_click():
    global click_received
    click_received = False
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
        
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

def find_next_checkpoint_direction(targetCoords, coords):
    """
    Takes the current location and the target and returns the direction 
    towards the target.
    """
    y, x = (targetCoords[0] - coords[0]), (targetCoords[1] - coords[1])
    
    if abs(x) > abs(y):
        if x < 0:
            return (-1, 0)
        else:
            return (1, 0)
    else:
        if y < 0:
            return (0, -1)
        else:
            return (0, 1)
        
    
def is_valid(x, y, visited, n=6):
    """
    Checks that a move resulted in a position that is still within
    the game board boundaries and that it has not been visited 
    yet (no intersecting previously visited positions).
    """
    return 0 <= x < n and 0 <= y < n and (x, y) not in visited

def solve_puzzle(board, coords, moves, directionalPriority = True):
    """
    A depth-first search that explores as deep as it can before backtracking
    """
    N = board.shape[0]
    path = []
    visited = set()
    recursions = 0
    move_order = moves.keys()
    
    def backtrack(x, y, target, visited_count):
        nonlocal recursions
        nonlocal move_order
        
        # Tracking recursions
        recursions += 1
        path.append((x, y))
        visited.add((x, y))
        
        # Initial completion condition - visiting all 36 spaces
        if visited_count == N * N:
            return True
        
        """
        Checks if current space is the list of coords that cooresponds
        to the positions of the checkpoints. 
        """
        if (x, y) in coords:
            # If on a checkpoint space (in coords), then check it is 
            # the correct one - so visiting them in order
            idx = coords.index((x, y)) + 1
            #if not, backtrack (first iteration always passes)
            if idx != target:
                path.pop()
                visited.remove((x, y))
                return False
            
            # Special check if on 8 - if its not the last tile hit,
            # backtrack.
            if (idx == 8 and 
                target == 8 and 
                visited_count != N * N
                ):
                path.pop()
                visited.remove((x, y))
                return False
            
            # If all checks passed, set the next target
            target += 1
            
        # Find and prioritize the direction of the next target
        # Only updates once after each checkpoint hit if enabled
        if target <= len(coords) and directionalPriority:
            tx, ty = coords[target-1]
            def dist(m):
                dx, dy = m
                return abs((x+dx)- tx) + abs((y+dy) - ty) # Manhattan
            move_order = sorted(move_order, key = dist)
            
        # Finally, make a move
        # Loops through each move (4) in the move set and checks
        # whether it is valid. If it is, it recurses again.
        for dx, dy in move_order:
            # Applies that move to the current coordinates
            nx, ny = x + dx, y + dy
            # Checks if it is a valid move (on board, not visited yet)
            if is_valid(nx, ny, visited, N):
                # if it is, it recurses another level with new coords
                if backtrack(nx, ny, target, visited_count + 1):
                    return True
        
        # if no valid move was found, it backtracks
        path.pop()
        visited.remove((x, y))
        return False
    
    start = coords[0]
    
    if backtrack(start[0], start[1], 1, 1):
        print("Solved!")
        return path, recursions
    
    print("No Solution")
    return None, recursions

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

# init listener thread for mouse listener globally because im shit at python
listener = mouse.Listener(on_click = on_click)



def main():
    check_environment()
    # Move set
    moves = {
        (-1, 0): "LEFT",
        (1, 0): "RIGHT",
        (0, -1): "UP",
        (0, 1): "DOWN"
    }
    # Prioritize direction of next checkpoint
    directionalPriority = False
    
    print("Click the window containing the puzzle... ")
    wait_for_click()
    time.sleep(1)
    
    i = 0
    try:
        while True:
            
            if i > 0:
                print("\nNew Game... ")
                wait_for_click()
                time.sleep(1)
                
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
            print(f"\n{displayBoard}\n")
            
            # Time solving the path
            startTime = time.time()
            solution, recursions = solve_puzzle(
                board,
                data.grid_locations, 
                moves,
                directionalPriority
                )
            endTime = time.time()
            elapsedTime = endTime - startTime
            
            print(f"Directional Priority: {directionalPriority}")
            print(f"Time to solve: {elapsedTime:.3f}s")
            print(f"Total recursions: {recursions}")
            
            if solution is not None:
                data.grid_to_pixels(solution).get_absolute_coords()
                complete_puzzle(data.pixel_coords)
                
            else:
                raise Exception("Lines already drawn - refresh the puzzle.\n If you keep getting this error, resize the window.")
                
            i+=1
        
    except KeyboardInterrupt:
        print("Exiting... ")
    
if __name__ == "__main__":
    main()