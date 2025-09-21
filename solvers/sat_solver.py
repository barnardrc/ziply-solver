# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 14:22:03 2025

@author: barna
"""
import numpy as np
from pysat.formula import CNF
from pysat.solvers import Solver

# Takes the board and returns the coordinates of checkpoints in order
def get_ordered_checkpoints(board):
    ordered_checkpoints = []
    num_checkpoints = len(np.where(board > 0)[0])
    for i in range(num_checkpoints):
        cp_loc = (np.where(board == i+1))
        r, c = cp_loc[0], cp_loc[1]
        cp = list(zip(r, c))[0]
        ordered_checkpoints.append(cp)
    return ordered_checkpoints


def solve_puzzle(board = None, *args):

    ROW = board.shape[0]
    COL = board.shape[1]
    
    
    #init formula
    formula = CNF()
    
    # List of all coords on board
    all_coords = []
    for r in range(ROW):
        for c in range(COL):
            all_coords.append((r, c))
            
    # Total time steps required to fill the board
    total_time_steps = COL * ROW
    
    # ordered checkpoint list
    checkpoints = get_ordered_checkpoints(board)
    
    # Var map:
    var_map = {}
    reverse_map = {}
    counter = 1
    for t in range(total_time_steps):
        for r in range(ROW):
            for c in range(COL):
                var_map[(r, c, t)] = counter
                reverse_map[counter] = (r, c, t)
                counter += 1
    
    # Coding the first constraint: Spaces can only be moved in 1 unit manhattan
    # distance
    for t in range(total_time_steps - 1): #iter through time steps
    # iter through all coords
        for coord1 in all_coords:
            for coord2 in all_coords:
                # Check for non-adjacent squares
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) != 1:
                    # This is every pair of variables from map that 
                    # cannot exist together because they are not adjacent
                    # between time consecutive time steps
                    var1 = var_map[(coord1[0], coord1[1], t)]
                    var2 = var_map[(coord2[0], coord2[1], t + 1)]
                    
                    # Add the logical clause to the CNF object
                    # This says: "The path cannot be at coord1 at time t AND at coord2 at time t+1"
                    
                    # NOT var1 OR NOT var2
                    formula.append([-var1, -var2])       
    # Second constraint: at least one and at most one (ALO and AMO) space must
    # be occupied for a single time step.
    # at least one:
    for t in range(total_time_steps):
        formula.append([var_map[(r, c, t)] for r in range(ROW) for c in range(COL)])
    
    # at most one:
    for t in range(total_time_steps): # loop all vars at each time step
        all_vars_at_time_t = [var_map[(r, c, t)] for r in range(ROW) for c in range(COL)]
        
        for i in range(len(all_vars_at_time_t)): 
            for j in range(i + 1, len(all_vars_at_time_t)):
                var1 = all_vars_at_time_t[i]
                var2 = all_vars_at_time_t[j]
                
                # var1 and var2 cannot be true at the same time.
                formula.append([-var1, -var2])
    
    # Third constraint: Checkpoints must be visited in order, so
    # cp1 < cp2, ... < ..., < cpn
    for i in range(len(checkpoints) - 1):
        cp1_loc = checkpoints[i]
        cp2_loc = checkpoints[i + 1]
        
        # Constraint: If cp2 is visited at time t, cp1 must be visited at a time < t
        for t2 in range(total_time_steps):
            vars_to_precede = []
            for t1 in range(t2):
                vars_to_precede.append(var_map[cp1_loc[0], cp1_loc[1], t1])
            
            clause = [-var_map[cp2_loc[0], cp2_loc[1], t2]] + vars_to_precede
            formula.append(clause)
    
    # Fourth contraint: spaces can be visited only once for all time steps.
    for r in range(ROW):
        for c in range(COL):
            # Get all variables for this specific square (r, c)
            vars_for_square = [var_map[(r, c, t)] for t in range(total_time_steps)]
    
            # Add a clause for every pair of variables, ensuring only one can be true
            for t1 in range(len(vars_for_square)):
                for t2 in range(t1 + 1, len(vars_for_square)):
                    var1 = vars_for_square[t1]
                    var2 = vars_for_square[t2]
                    formula.append([-var1, -var2])
    
    # Fifth constraint: where the model must start
    start_loc = checkpoints[0]
    if start_loc is not None:
        formula.append([var_map[start_loc[0], start_loc[1], 0]])
    else:
        print("Starting checkpoint (1) not found.")
        return None
    
    # Sixth constraint: where the model must end
    last_checkpoint_loc = checkpoints[-1]
    last_time_step = total_time_steps - 1
    if last_checkpoint_loc is not None:
        # Only one variable corresponds to the path being at the last checkpoint 
        # at the last time step. 
        last_var = var_map[(last_checkpoint_loc[0], last_checkpoint_loc[1], last_time_step)]
        
        # Single clause. last_var must evaluate to true
        formula.append([last_var])
        
    #print(len(f"Total Clauses: {formula.clauses}"))
    
    
    with Solver(bootstrap_with=formula.clauses) as s:
        if s.solve():
            model = s.get_model()
            
            solution_path = [None] * total_time_steps
            for var in model:
                # Ensure it uses only the positive integers from model
                if var > 0:
                    # Map solution back to coordinates in var map
                    r, c, t = reverse_map[var]
                    # Generate solution path for return
                    solution_path[t] = (r, c)
            
            print("Solution found!")
            return solution_path
        
        else:
            print("No Solution")
            return None
        
if __name__ == "__main__":
    solve_puzzle()