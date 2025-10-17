# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:54:23 2025

A constraint based search for finding a valid path that runs through all checkpoints in order.
* Does not solve the ziply puzzle.

@author: barna
"""

from solvers.dependents.dependents import (
    get_closed_coords as gcc, 
    get_loc, 
    is_adjacent, 
    is_within_board
    )
from solvers.dependents import flipper

import numpy as np
from numpy import random
import numpy as np
import heapq
import time

class VariableHandler:
    def __init__(self):
        self.counter = 0
        
    def add_one(self):
        self.counter += 1
    
    def get_counter(self):
        return self.counter

class PathSegment:
    def __init__(self, src, dest, target_id):
        self.src = src # src
        self.dest = dest # dest
        self.target_id = target_id # Checkpoint destination ID
        
        self.coordinates = [] # Coordinates of path
        self.status = "PENDING" # status: PENDING, BLOCKED, SOLVED
        
        self.blocking_paths = set() # target_ids that are blocking this path
        self.blacklist = [] # target_ids that this path has already deleted
        
        self.constraints = set()
        
    def add_constraint(self, constraining_coord):
        self.constraints.add(constraining_coord)
    
    def update_coordinates(self, path_coordinates):
        if self.target_id != 2:
            self.coordinates = path_coordinates[1:]
            
        else:
            self.coordinates = path_coordinates[:]
    
class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        self.time = 0
        
def is_unblocked(grid, row, col, next_checkpoint_value, constraints):
    """
    Checks if a grid cell is a valid move.
    0 is a valid space. A checkpoint is only valid if it's the next destination.
    """
    cell_value = grid[row][col]
    
    if constraints is not None and (row, col) in constraints:
        return False
    
    # It's an unblocked space if its value is 0
    if cell_value == 0:
        return True
    
    # It's an unblocked space if its value is the next checkpoint
    if cell_value == next_checkpoint_value and next_checkpoint_value is not None:
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
        
def a_star_search(grid, src, dest, next_checkpoint_value, constraints, 
                  counter, directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]):
    
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
        
        for direc in directions:
            new_i = i + direc[0]
            new_j = j + direc[1]
            counter.add_one()
            
            if (is_valid(new_i, new_j) and
                is_unblocked(grid, new_i, new_j, next_checkpoint_value, 
                             constraints) and not 
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

# Orders a dictionary by key value. Expepcts keys to be integers.
def order_dict(xdict):
    return dict(sorted(xdict.items(), key=lambda item: int(item[0])))

# Makes each path segment in the dictionary complete, going from one
# checkpoint to the next, inclusive.
def prepend_prev_last_item(xdict):
    for i in range(2, len(xdict) + 1):
        first_value = xdict[i][-1]
        second_value = xdict[i+1]
        
        second_value.insert(0, first_value)

def calculate_total_cost(dict_of_paths):
    return sum([len(path) for path in dict_of_paths.values()])

def find_conflict(path_segments):
    all_coords = {}
    for path_id, segment in path_segments.items():
        for coord in segment:
            if coord not in all_coords:
                all_coords[coord] = []
            all_coords[coord].append(path_id)
            
            if len(all_coords[coord]) > 1:
                conflicting_path_ids = tuple(all_coords[coord])
                return {"location": coord, "path_ids": conflicting_path_ids}
                
    return None

def create_branch(board, path_segments, current_node, conflicting_coord, 
                  path_id, counter):
    
    new_constraints = current_node["constraints"].copy()
    new_constraints[path_id] = new_constraints.get(path_id, set()) | {conflicting_coord}
    new_paths = current_node["paths"].copy()
    
    new_path = a_star_search(board, path_segments[path_id].src,
                             path_segments[path_id].dest, path_id,
                             new_constraints[path_id], counter)
    
    if new_path is not None:
        if path_id != 2:
            new_paths[path_id] = new_path[1:]
        else:
            new_paths[path_id] = new_path[:]
            
        cost = calculate_total_cost(new_paths)
        node = {"cost": cost, "constraints": new_constraints, "paths": new_paths}
        
        num_constraints = sum(len(c) for c in new_constraints.values())
        counter.add_one()
        
        return (cost, num_constraints, counter.get_counter(), node)
    
    else:
        return None

def return_solved(board, current_node, iteration, do_flipper):
    target_to_path = order_dict(current_node["paths"])
    
    if do_flipper:
        # Prepend for flipper (expects full paths all coords inclusive)
        prepend_prev_last_item(target_to_path)
        full_path = flipper.flip(board, target_to_path)
        
    else:
        full_path = gcc(target_to_path)
    
    full_path = gcc(target_to_path)
    return full_path, iteration

def solve_puzzle(board, simulationLength = None):
    do_flipper = True
    counter = VariableHandler()
    
    path_segments = {}
    total_checkpoints  = np.count_nonzero(board > 0)
    for i in range(2, total_checkpoints + 1):
        start_node = get_loc(board, i - 1)
        end_node = get_loc(board, i)
        path_segments[i] = PathSegment(start_node, end_node, target_id=i)
    
    for path, path_object in path_segments.items():
        path_coordinates = a_star_search(
            board,
            path_object.src,
            path_object.dest,
            path,
            path_object.constraints,
            counter
            )
        
        path_object.update_coordinates(path_coordinates)
        
    initial_paths = {path_object.target_id: path_object.coordinates for _, path_object in path_segments.items()}
    initial_cost = calculate_total_cost(initial_paths)
    initial_node = {
        "cost": initial_cost,
        "constraints": {},
        "paths": initial_paths
    }
    
    priority_queue = [(initial_node["cost"], 0, 0, initial_node)]
    iteration = 0
    while priority_queue:
        iteration += 1
        current_cost, _, _, current_node = heapq.heappop(priority_queue)
        
        conflict = find_conflict(current_node["paths"])
        
        if conflict is None:
            return return_solved(board, current_node, iteration, do_flipper)
        
        conflicting_coord = conflict['location']
        path_A_id, path_B_id = conflict['path_ids']
        
        branch_A = create_branch(board, path_segments, current_node, 
                               conflicting_coord, path_A_id, counter)
        
        branch_B = create_branch(board, path_segments, current_node, 
                               conflicting_coord, path_B_id, counter)

        if branch_A is not None:
            heapq.heappush(priority_queue, branch_A)

        if branch_B is not None:
            heapq.heappush(priority_queue, branch_B)
        
if __name__ == "__main__":
    solve_puzzle()