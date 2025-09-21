# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 14:33:46 2025

@author: barna
"""
import numpy as np
import random
from sat_solver import solve_puzzle


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

def main():
    H = 6
    W = 6
    for i in range(100):
        num_checkpoints = H*W // 2
        x = generate_random_board(H, W, num_checkpoints)
        print(x)
        solve_puzzle(x)
        
    
main()