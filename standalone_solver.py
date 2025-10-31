# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:30:10 2025

@author: barna
"""

# @!! set the solver you would like to use following solvers in THIS import statement !!@
# Ex: solvers.cbs, solvers.backwards_dfs - Additional solvers in solvers directory
from solvers.cbs import solve_puzzle
from custom_boards.npzchecker import get_boards
import numpy as np
import time

def is_complete(board, solution):
    H, W = board.shape
    return len(solution) == H*W

# Will check every board in the desired dimensions from the target .npz file
# get the amount of
# a) odd spaces remaining
# b) failed boards
# No boards should fail for existing npz files
# This script can be customized to target a specific parameter, i.e., 
#       - amount of boards an incomplete solver fails on
#       - spaces remaining
#       - spaces filled
#       - time it takes to solve
def check_boards_in_sequence():
    odd_spaces_boards = []
    failed_boards = []
    even_spaces_remaining_list = []
    
    all_boards = get_boards(dims_to_check = '6x6')
    keys = list(all_boards.keys())

    for key in keys:
        print(f"Board: {key}")
        board = all_boards[key]
        title = f'{key}'
        
        solution, iter_num = solve_puzzle(board)
        if solution is None:
            failed_boards.append(key)
            break
        
        length = len(solution)
        openSpaces = 36 - length
        isEven = openSpaces % 2 == 0
        if isEven:
            even_spaces_remaining_list.append(key)
        if not isEven:
            odd_spaces_boards.append(key)
        
    print(f"Amount of boards with odd # of spaces remaining: {len(odd_spaces_boards)}")
    print(f"Amount of failed boards: {len(failed_boards)}")
    print(f"Amount with even spaces remaining: {len(even_spaces_remaining_list)}")
    print("(Failed Boards:)")
    for key in failed_boards:
        print(all_boards[key])
        
    print("(Boards With Odd Spaces Remaining:)")
    for key in odd_spaces_boards:
        print(all_boards[key])
    
# Check a specific board by key in the desired dimensions. Good for getting a 
# solution to a specific board.
def check_board():
    # Set the dimension .npz file to check, then run!
    # It will provide a list of (solvable) boards in the destination, then
    # ask you for the particular board you want to run the solver on.
    all_boards = get_boards(dims_to_check = '9x9')
    print("Boards available:")
    for board in all_boards:
        print(board)
    
    keys = list(all_boards.keys())
    target_arr = input("\nInput target board: ")
    target_board = "arr_" + target_arr
    if target_board not in keys:
        print(f"{target_board} does not exist!")
    else:
        print(f"\nSolving {target_board}...")
        start_time = time.time()
        solution = solve_puzzle(all_boards[target_board])
        elapsed_time = time.time() - start_time
        
        print(f"It took {elapsed_time} to solve this board.")
        print(f"Solution Length: {len(solution[0])}")
        print(f"Solution: {solution[0]}")
    
if __name__ == "__main__":
    check_board()