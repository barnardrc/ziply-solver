# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:31:17 2025

@author: barna
"""
import numpy as np
import numbers

def get_checkpoint_vals_to_coords(board):
    return sorted([(board[r, c], (r, c)) for r,c in np.ndindex(board.shape) if board[r, c] > 0])

# Returns coordinates ordered
def get_ordered_checkpoints(board):
    ordered_checkpoints = []
    num_checkpoints = len(np.where(board > 0)[0])
    for i in range(num_checkpoints):
        cp_loc = (np.where(board == i+1))
        r, c = cp_loc[0], cp_loc[1]
        cp = list(zip(r, c))[0]

        ordered_checkpoints.append(cp)

    return ordered_checkpoints

def get_closed_coords(dict_of_coords):
    """

    """
    return [coord for coord_list in dict_of_coords.values() for coord in coord_list]

        
def get_loc(board, target):
    """

    """
    
    if isinstance(target, numbers.Integral):
        coords = np.where(board == target)
        row = coords[0][0]
        col = coords[1][0]
        
        return (row, col)
    else:
        return None

def is_adjacent(coord1, coord2, target_row = None, target_col = None):
    """
    
    """
    if isinstance(coord1, numbers.Integral) and isinstance(coord2, numbers.Integral):
        row = coord1
        col = coord1
        return (abs(row - target_row) == 1 and col == target_col) or \
               (abs(col - target_col) == 1 and row == target_row)
    elif isinstance(coord1, (tuple, list, set)) and isinstance(coord2, (tuple, list, set)):
        row = coord1[0]
        col = coord1[1]
        target_row = coord2[0]
        target_col = coord2[1]
    
        
        return (abs(row - target_row) == 1 and col == target_col) or \
               (abs(col - target_col) == 1 and row == target_row)
               
    else:
        raise ValueError("is_adjacent expects integer or tuple.")
    
    
def _get_sequential_pairs(xlist):
    new_list = []
    for i in range(len(xlist) - 1):
        temp_list = xlist[i:i+2]
        new_list.append(temp_list)
    
    return new_list

def is_continuous(path_segment, row, col, target_row, target_col):
    pass

def is_within_board(x, y, visited, n=6):
    """
    Checks that a move resulted in a position that is still within
    the game board boundaries
    """
    return 0 <= x < n and 0 <= y < n