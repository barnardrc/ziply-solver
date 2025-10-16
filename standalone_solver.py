# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:30:10 2025

@author: barna
"""

from solvers.in_progress.a_star_plus_cbs import solve_puzzle
from custom_boards.npzchecker import get_boards
import numpy as np

def is_complete(board, solution):
    H, W = board.shape
    return len(solution) == H*W

def check_boards_in_sequence():
    incomplete_list = []
    iteration_list = []
    failed_boards = []
    
    all_boards = get_boards(dims_to_check = '6x6')
    keys = list(all_boards.keys())

    for key in keys:
        try:
            print(f"Board: {key}")
            board = all_boards[key]
            title = f'{key}'
            
            solution, iter_num = solve_puzzle(board)
            
            
            if not is_complete(board, solution):
                incomplete_list.append(key)
            
            iteration_list.append(iter_num)
            
        except:
            failed_boards.append(key)
        
    max_iterations = max(iteration_list)
    print(f"Boards not solved by flipper: {incomplete_list}")
    print(f"Maximum iterations for A* to find a path: {max_iterations}")
    print(f"Boards where A* found no full path: {failed_boards}")
    for board in failed_boards:
        print(f"board {board}")
        print(all_boards[board])
    
            
if __name__ == "__main__":
    check_boards_in_sequence()
    