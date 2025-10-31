# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 14:33:46 2025

!! README
This script is equipped to solve a singular, passed board if desired.

Its main function is to 
1. Generate random boards 
2. Attempt to solve them with the desired solver (set in the import statement)
3. Save solvable boards to a .npz file for lookup

If desired, the script could be adjusted to save all boards.

@author: barna
"""
import numpy as np
import random

# @!! CHANGE SOLVER HERE - ADDIT. SOLVERS LOCATED IN /solvers !!@
from solvers.cbs import solve_puzzle

import time
import re

def generate_random_board(height: int, width: int, num_checkpoints: int):
    
    if num_checkpoints > height * width:
        raise ValueError("Number of checkpoints cannot exceed board size.")
    
    # Create an empty board
    board = np.zeros((height, width), dtype=int)
    
    # Create a list of all possible coordinates
    all_coords = [(r, c) for r in range(height) for c in range(width)]
    
    # Shuffle the coordinates to get random positions
    random.shuffle(all_coords)
    
    # Place the checkpoints
    checkpoint_coords = all_coords[:num_checkpoints]
    for i, (r, c) in enumerate(checkpoint_coords):
        board[r, c] = i + 1
    
    return board


def get_file_name(path, H, W, num_checkpoints, cbs = False):
    if cbs:
        return path + f"{H}x{W}_cp-{num_checkpoints}_cbs"
    
    else:
        return path + f"{H}x{W}_cp-{num_checkpoints}"


def main():
    folder_path = "custom_boards/"
    board = np.array(
[[0, 0, 0, 0, 2],
 [0, 0, 0, 0, 0],
 [0, 0, 4, 0, 0],
 [0, 3, 0, 0, 0],
 [1, 0, 0, 0, 5]]
    )
    
    # Generate random board: True
    # Use provided board as array from above: False
    random_board = True
    
    # If using CBS on the boards, set this param to true so it gets added to the
    # npz file name (A board solved with CBS is not guaranteed to have a solution)
    cbs = True
    
    # If you would like to set a target amount of boards to solve before
    # exiting, do so here.
    solvable_amount_target = float('inf')
    
    if random_board:
        
        # Handle user input
        try:
            print("Ctrl + C to cancel")
            # Dims of boards to gen
            dimensions = input("Input desired dimensions 'HxW': ")
            dims = re.split('x|X|,| ', dimensions)
            if len(dims) != 2:
                raise ValueError("Ensure H and W are separated by an accepted\
                                 separator: ['x', 'X', ',', ' ']")
            else:
                H = int(dims[0])
                W = int(dims[1])
        except:
            raise TypeError("Dimesnions must be integers.")
        
        try:
            # Num checkpoints to put on board
            num_checkpoints = int(input("Input desired checkpoints <int>: "))
            # Amount of boards to attempt
            amount_boards = int(input("Input amount of boards to check: "))
            
        except:
            raise TypeError("The number of checkpoints and amount of boards\
                            must be integers.")
        
        file_name = get_file_name(folder_path, H, W, num_checkpoints, cbs)
        print(file_name)
        
        try:
            with np.load(file_name + '.npz', allow_pickle=True) as data:
                # Loading arrays into a set for easy uniqueness checking
                stored_arrays_set = set(tuple(arr.flatten()) for arr in data.values())
                
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty set
            stored_arrays_set = set()
            print("No existing .npz file found. Starting with an empty dataset.")
        
        # Feedback
        print(f"\nDimensions: {H}x{W}")
        print(f"Number of Checkpoints: {num_checkpoints}")
        print(f"Amount of boards: {amount_boards}\n")
        
        # Begin
        start_time = time.time()
        successes = 0
        for i in range(amount_boards):
            boards_left = amount_boards - i + 1
            if solvable_amount_target == 0:
                print("Target amount of boards found!")
                break
            
            if i % 100 == 0:
                print(f"Boards found: {successes}")
                print(f"Boards left to check: {boards_left}")
                
            board = generate_random_board(H, W, num_checkpoints)
            print(f"Board #{i+1}: ", end = '')
            solution = solve_puzzle(board)
            
            if solution is not None:
                open_spaces_left = (H*W) - len(solution[0])
                open_spaces_are_even = open_spaces_left % 2 == 0
                print(f"Open Spaces: {open_spaces_left}")
                if cbs and not open_spaces_are_even:
                    print(np.array2string(board, separator=', '))
                    print("Skipping board... ")
                    break
                successes += 1
                # Convert to 2 string for easy copy/paste
                print(np.array2string(board, separator=', '))
                # Convert to tuple for storing
                new_board = tuple(board.flatten())
                
                # Check if board already exists
                if new_board not in stored_arrays_set:
                    print("New unique array found. Adding to the set.")
                    stored_arrays_set.add(new_board)
                    solvable_amount_target -= 1
                
                else:
                    print("Array already exists. Discarding.")
                    
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime to solve {amount_boards} boards: {elapsed_time:.3f}s")
        
        # Convert the set of tuples back to a list of arrays for saving
        arrays_to_save = [np.array(t).reshape(H, W) for t in stored_arrays_set]
        
        # Save the updated list of arrays to the .npz file
        np.savez(file_name, *arrays_to_save)
        print(f"Updated data saved to {file_name}.npz")
    
    # If passing your own board, just solve the puzzle
    else:
        print(board)
        start_time = time.time()
        solution = solve_puzzle(board)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime to solve: {elapsed_time:.3f}s")
        print(solution)

        
main()