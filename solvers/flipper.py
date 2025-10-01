# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:33:49 2025

Takes an already drawn path on a target board and attempts to flip the rest of
the squares to visited in the checkpoint order it received.

@author: barna
"""

from solvers.dependents import Helper
import numpy as np

def get_open_coords(closed_board):
    open_coords = list(zip(*np.where(closed_board == 0)))
    return open_coords

def get_board(board = None, dict_of_coords = None, closed_coords = None):
    closed_board = np.zeros_like(board)
    
    rows, cols = zip(*closed_coords)
    closed_board[rows, cols] = 1
    
    #print(closed_board.T)
    return closed_board

def is_valid(x, y, visited, n=6):
    """
    Checks that a move resulted in a position that is still within
    the game board boundaries
    """
    return 0 <= x < n and 0 <= y < n


def is_adjacent(row, col, target_row, target_col):
    return (abs(row - target_row) == 1 and col == target_col) or \
           (abs(col - target_col) == 1 and row == target_row)

def insert_flipped(target_list, src, flip_list):
    insert_idx = target_list.index(src) + 1
    target_list[insert_idx:insert_idx] = flip_list

# Takes the path to a specified checkpoint and finds the open coords that 
# are adjacent.
def flip_adjacent_open(open_coords = None,
                       dict_of_coords = None,
                       cp = 2
                       ):
    
    local_open = open_coords[:]
    path_segment = dict_of_coords[cp]
    
    adjacent_coords = []
    flip_list = []
    found = None
    
    for i, (x1, y1) in enumerate(path_segment):
        
        for (x2, y2) in local_open:
            if is_adjacent(x1, y1, x2, y2):
                adjacent_coords.append((x2, y2))
                if len(adjacent_coords) > 1:
                    
                    for (a1, b1) in adjacent_coords:
                        
                        for (a2, b2) in adjacent_coords:
                            if is_adjacent(a1, b1, a2, b2):
                                flip_list.append((a1, b1))
                                src = path_segment[i-1]
                                found = True
                                break
                    if found:
                        break
        if found:
            break
    
    if found:
        #print(f"From space {src}, fill {flip_list[0]} then {flip_list[1]} before continuing.")
        insert_flipped(path_segment, src, flip_list)
        dict_of_coords[cp] = path_segment
        return True
    
    else:
        # Once its done flipping for a particular path, cut out the first
        # coord to prevent duplicates
        if cp != 2:
            dict_of_coords[cp] = path_segment[1:]
            print(path_segment[1:])
        #print(f"No changes made from checkpoint {cp}.")
        return False
    #print(dict_of_coords[cp])
    
    
    
def flip(board, target_to_path):
    
    total_checkpoints = np.sum(board > 0)
    start = 2
    
    counter = start
    while counter < total_checkpoints + 1:
        # Get all filled coords
        closed_coords = Helper.get_closed_coords(target_to_path)
        
        # Create a mask that represents all closed coords
        closed_board = get_board(board, target_to_path, closed_coords)
        
        # Get unvisited spaces based off of that board 
        # This removes having to check for spaces within the game board
        open_coords = get_open_coords(closed_board)
        
        
        if not flip_adjacent_open(open_coords, target_to_path, cp = counter):
            counter += 1
            
    return Helper.get_closed_coords(target_to_path)

if __name__ == "__main__":
    flip()