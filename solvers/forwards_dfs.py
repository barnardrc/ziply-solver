# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:01:10 2025

solvers.py

Backtracking DFS Search algorithm

@author: barna
"""
import numpy as np
from solvers.dependents.dependents import get_ordered_checkpoints

def is_valid_backtrack_dfs(x, y, visited, n=6):
    """
    Checks that a move resulted in a position that is still within
    the game board boundaries and that it has not been visited
    yet (no intersecting previously visited positions).
    """
    return 0 <= x < n and 0 <= y < n and (x, y) not in visited

def solve_puzzle(board, simulationLength = None
                  ):
    coords = get_ordered_checkpoints(board)
    num_checkpoints = len(coords)
    
    # Move set
    moves = {
        (-1, 0): "LEFT",
        (1, 0): "RIGHT",
        (0, -1): "UP",
        (0, 1): "DOWN"
    }
    
    """
    A depth-first search that explores as deep as it can before backtracking
    """
    N = board.shape[0]
    path = []
    visited = set()
    visited_all = []
    recursions = 0
    move_order = moves.keys()
    
    def backtrack(x, y, target, visited_count):
        nonlocal recursions
        nonlocal move_order
        nonlocal num_checkpoints
        
        # Tracking recursions
        recursions += 1
        path.append((x, y))
        visited.add((x, y))
        
        # How many coords to save for playback of the algorithm
        if simulationLength:
            if recursions <= simulationLength:
                visited_all.append((y, x))
        
        # Initial completion condition - visiting all 36 spaces
        if visited_count == N * N:
            return True
        
        """
        Checks if current space is in the list of coords that cooresponds
        to the positions of the checkpoints. 
        """
        if (x, y) in coords:
            # If on a checkpoint space (in coords), then check it is 
            # the correct one - so visiting them in order
            idx = coords.index((x, y)) + 1
            #if not, backtrack (first iteration always passes)
            if idx != target:
                path.pop()
                visited.remove((x, y))
                return False
            
            # Special check if on 8 - if its not the last tile hit,
            # backtrack.
            if (idx == num_checkpoints and
                target == num_checkpoints and
                visited_count != N * N
                ):
                path.pop()
                visited.remove((x, y))
                return False
            
            # If all checks passed, set the next target
            target += 1
            
        # Finally, make a move
        # Loops through each move (4) in the move set and checks
        # whether it is valid. If it is, it recurses again.
        for dx, dy in move_order:
            # Applies that move to the current coordinates
            nx, ny = x + dx, y + dy
            # Checks if it is a valid move (on board, not visited yet)
            if is_valid_backtrack_dfs(nx, ny, visited, N):
                # if it is, it recurses another level with new coords
                if backtrack(nx, ny, target, visited_count + 1):
                    return True
        
        # if no valid move was found, it backtracks
        path.pop()
        visited.remove((x, y))
        return False
    
    start = coords[0]
    
    if backtrack(start[0], start[1], 1, 1):
        print("Solved!")
        return path
    
    print("No Solution")
    
    return None