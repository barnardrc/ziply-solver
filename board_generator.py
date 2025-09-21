# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 14:33:46 2025

@author: barna
"""
import numpy as np
import random
from solvers.sat_solver import solve_puzzle
import time

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


def get_file_name(path, H, W):
    return path + f"{H}x{W}"


def main():
    board = np.array(
[[ 0,  0,  0,  0,  4,  0,  0],
 [ 0,  2,  0,  0,  0,  0,  0],
 [ 0,  3,  1,  0,  0,  7,  0],
 [ 0,  0,  0,  0,  0,  0,  0],
 [ 0,  9,  0,  0,  6,  0,  0],
 [ 0, 10,  0,  0,  0,  8,  0],
 [ 0,  0,  0,  0,  0,  0,  5]]
    )
    random_board = True
    amount_boards = 1000
    
    elapsed_time = None
    
    H = 8
    W = 8
    num_checkpoints = 12
    
    folder_path = "custom_boards/"
    file_name = get_file_name(folder_path, H, W)
    print(file_name)
    
    try:
        with np.load(file_name + '.npz', allow_pickle=True) as data:
            # Loading arrays into a set for easy uniqueness checking
            stored_arrays_set = set(tuple(arr.flatten()) for arr in data.values())
            
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty set
        stored_arrays_set = set()
        print("No existing .npz file found. Starting with an empty dataset.")

    
    if random_board:
        print(f"\nDimensions: {H}x{W}")
        print(f"Number of Checkpoints: {num_checkpoints}\n")
        start_time = time.time()
        for i in range(amount_boards):
            
            board = generate_random_board(H, W, num_checkpoints)
            print(f"Board #{i+1}: ", end = '')
            solution = solve_puzzle(board)
            
            if solution is not None:
                # Convert to 2 string for easy copy/paste
                print(np.array2string(board, separator=', '))
                # Convert to tuple for storing
                new_board = tuple(board.flatten())
                
                # Check if board already exists
                if new_board not in stored_arrays_set:
                    print("New unique array found. Adding to the set.")
                    stored_arrays_set.add(new_board)
                
                else:
                    print("Array already exists. Discarding.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime to solve {amount_boards} boards: {elapsed_time:.3f}s")
    # If passing your own board, just solve the puzzle
    else:
        print(board)
        start_time = time.time()
        solution = solve_puzzle(board)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTime to solve: {elapsed_time:.3f}s")
        #print(solution)
        
    # Convert the set of tuples back to a list of arrays for saving
    arrays_to_save = [np.array(t).reshape(H, W) for t in stored_arrays_set]
    
    # Save the updated list of arrays to the .npz file
    np.savez(file_name, *arrays_to_save)
    print(f"Updated data saved to {file_name}.npz")
        
main()