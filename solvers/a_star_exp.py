# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:03:47 2025

Manhattan distance A* Search Algorithm with checkpoints

@author: barna
"""

from solvers.dependents import Helper
import numpy as np
import heapq
class VariableHandler:
    def __init__(self):
        self.counter = 0
        
    def add_one(self):
        self.counter += 1
    
    def get_counter(self):
        return self.counter

class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        self.time = 0
        
def is_unblocked(grid, row, col, next_checkpoint_value, visited_coords):
    """
    Checks if a grid cell is a valid move.
    0 is a valid space. A checkpoint is only valid if it's the next destination.
    """
    cell_value = grid[row][col]
    
    if(row, col) in visited_coords:
        return False
    
    # It's an unblocked space if its value is 0
    if cell_value == 0:
        return True
    
    # It's an unblocked space if its value is the next checkpoint
    if cell_value == next_checkpoint_value:
        return True
    
    # Any other checkpoint is blocked
    return False

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def calculate_h_value(row, col, dest):
    return abs(row - dest[0]) + abs(col - dest[1])

def trace_path(cell_details, dest):
    #print("Running trace_path... ")
    path = []
    row = dest[0]
    col = dest[1]
    
    # Trace the path backwards from the destination to the source
    while not (cell_details[row][col].parent_i == row and
                cell_details[row][col].parent_j == col):
        
        path.append((row, col))
        
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        
        row = temp_row
        col = temp_col
    
    # Append the source node to the path
    path.append((row, col))
    
    # Reverse the list to get the path from source to destination
    path.reverse()
    
    #print("trace_path finished")
    return path
        
def a_star_search(grid, src, dest, next_checkpoint_value, visited_coords, 
                  counter):
    
    #print("Running search... ")
    ROW = grid.shape[0]
    COL = grid.shape[1]

    def is_valid(row, col):
        return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return None

    if is_destination(src[0], src[1], dest):
        print("Already at destination")
        return [src]
    
    # init closed list. All False because none are visited in the beginning
    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    
    # Init cell objects for each cell on the board.
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]
    
    # declare i, j - the source cell coordinates
    i, j = src
    # init costs
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    # declare parent coords
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j
    
    # init open list
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))
    
    while len(open_list) > 0:
        # Remove and return the open list from the heap
        p = heapq.heappop(open_list)
        
        i, j = p[1], p[2]
        closed_list[i][j] = True
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for direc in directions:
            new_i = i + direc[0]
            new_j = j + direc[1]
            counter.add_one()
            
            if (is_valid(new_i, new_j) and
                is_unblocked(grid, new_i, new_j, next_checkpoint_value, 
                             visited_coords) and not 
                closed_list[new_i][new_j]):
                
                if is_destination(new_i, new_j, dest):
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    return trace_path(cell_details, dest)
                
                else:
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new
                    
                    if (cell_details[new_i][new_j].f == float('inf') or
                        cell_details[new_i][new_j].f > f_new):
                        
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        
    return None

def get_target(target_dict, total_checkpoints, start = 2):
    missing_targets = []
    for i in range(start, total_checkpoints + 2):
        if i not in target_dict:
            missing_targets.append(i)
    
    return min(missing_targets)

def order_dict(xdict):
    return dict(sorted(xdict.items(), key=lambda item: int(item[0])))

def solve_puzzle(board, dummy_coords = None, simulationLength = None):
    counter = VariableHandler()
    visited_coords = set(Helper.get_loc(board, 1))
    total_checkpoints = np.sum(board > 0)
    target_to_path = {}
    
    # Starting location is checkpoint 1
    current_loc = Helper.get_loc(board, 1)
    
    if current_loc is None:
        print("Starting checkpoint not found.")
        return
        
    start = 2
    max_iter = 100
    
    
    #print(f"total checkpoints: {total_checkpoints}")
    target = get_target(target_to_path, total_checkpoints, start)
    iter_num = 0
    while target <= total_checkpoints and iter_num < max_iter:
        
        next_loc = Helper.get_loc(board, target)
        
        if next_loc is None:
            print(f"Checkpoint {target} not found.")
            
            
        print(f"Current Location: {target - 1}")
        print(f"Target Location: {target}")
        path_segment = a_star_search(board,
                                     current_loc,
                                     next_loc,
                                     target,
                                     visited_coords,
                                     counter,
                                     )
        
        if path_segment is not None:
            # Append the path segment, but skip the first node (current_loc)
            # to avoid duplicates
            if target != 2:
                target_to_path[target] = path_segment[1:]
            else:
                target_to_path[target] = path_segment[:]
                
            target = get_target(target_to_path, total_checkpoints, start)
            attempt = 0
            
        else:
            attempt += 1
            print()
            print(f"Failed to find a path to checkpoint {target}.")
            print()
            target_to_path = order_dict(target_to_path)
            print(f"Dict after ordering: {target_to_path}")
            try:
                del_idx = target-attempt
                deleted_path = target_to_path.pop(del_idx)
                
            except:
                target_to_path.clear()
            
                
            print(f"Deleted Path: {deleted_path}")
            
        # Get visited coords
        visited_coords = Helper.get_closed_coords(target_to_path)
        if len(visited_coords) == 0:
            print("Dict successfully cleared and visited_coords is empty.")
        #print(f"visited set: {visited_coords}")

        current_loc = Helper.get_loc(board, target - 1)
        iter_num += 1
        
    #print(target_to_path)
    loops = counter.get_counter()
    
    return Helper.get_closed_coords(order_dict(target_to_path))