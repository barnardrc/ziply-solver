# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 08:44:14 2025

@author: barna
"""

# Take a list and separate it into pairs, so [[(0,1), (0,0)], [(0, 0), (1, 0)]]
from solvers.dependents.dependents import is_adjacent, get_closed_coords as gcc


def get_sequential_pairs(xlist):
    for i in range(len(xlist) - 1):
        print(xlist[i:i+2])

def get_sequential_pairs(xlist):
    new_list = []
    for i in range(len(xlist) - 1):
        temp_list = xlist[i:i+2]
        new_list.append(temp_list)
    
    return new_list

# Gets coords adjacent to the path and uses its corresponding path as the key
def get_adjacent_coords(path_segment, open_coords):
    adjacent_coords = {}
    for (x1, y1) in (path_segment):
        temp_list = []
        for (x2, y2) in open_coords:
            if is_adjacent(x1, y1, x2, y2):
                temp_list.append((x2, y2))
                adjacent_coords[(x1, y1)] = temp_list
                
    return adjacent_coords

def find_first_flippable_pair(path, open_coords):
    """
    Finds the first flippable pair of coordinates based on a sequential path.

    A pair (C1, C2) is "flippable" for a path segment (S1, S2) if:
    1. (S1, S2) are sequential coordinates in the `path`.
    2. C1 is in `open_coords` and is adjacent to S1.
    3. C2 is in `open_coords` and is adjacent to S2.
    4. C1 and C2 are adjacent to each other.
    """
    # 1. Iterate through each sequential pair (segment) in the path
    print(path)
    for i in range(len(path) - 1):
        source1, source2 = path[i], path[i+1]

        # 2. Find all open coordinates adjacent to Source1 and Source2
        adj_to_source1 = [c for c in open_coords if is_adjacent(source1, c)]
        adj_to_source2 = [c for c in open_coords if is_adjacent(source2, c)]

        # If either has no adjacent coords, this segment can't have a flippable pair
        if not adj_to_source1 or not adj_to_source2:
            continue

        # 3. Check if any coordinate from the first list is adjacent to any from the second
        for coord1 in adj_to_source1:
            for coord2 in adj_to_source2:
                # 4. The final adjacency check
                if is_adjacent(coord1, coord2):
                    # Found the first flippable pair, return it and stop.
                    return {
                        "flippable_pair": (coord1, coord2),
                        "source_segment": (source1, source2)
                    }
                    
    
    # If the loop finishes without finding any, return None
    return None
    
def main():
    my_list = [(0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (2, 1), (2, 0), 
               (3, 0), (3, 1), (3, 2), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5), 
               (1, 4), (1, 3)
               ]
    
    open_coordinates = [
      (0, 0),
      (1, 0),
      (4, 0),
      (5, 0),
      (4, 1),
      (5, 1),
      (4, 2),
      (5, 2),
      (3, 3),
      (4, 3),
      (5, 3),
      (3, 4),
      (4, 4),
      (5, 4),
      (3, 5),
      (4, 5),
      (5, 5)
    ]
    
    #get_sequential_pairs(my_list)

    
    
    x = find_first_flippable_pair(my_list, open_coordinates)
    print(x)
    
main()