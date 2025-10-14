# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 08:44:14 2025

@author: barna
"""

# Take a list and separate it into pairs, so [[(0,1), (0,0)], [(0, 0), (1, 0)]]
from solvers.dependents import is_adjacent, get_closed_coords as gcc


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

def get_flippable_coords(src_to_adjacent_dict: dict):
    all_coords = gcc(src_to_adjacent_dict)
    for coords in src_to_adjacent_dict.values():
        for (x1, y1) in coords:
            print(f"Comparing coord {(x1, y1)}")
            for (x2, y2) in coords:
                print(f"With {(x2, y2)}")
                if is_adjacent(x1, y1, x2, y2):
                    print(f"Coord found: {(x1, y2), (x2, y2)}")
    
def main():
    my_list = [(2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (1, 2), (2, 2), (3, 2), 
               (4, 2), (4, 3), (3, 3), (3, 4), (3, 5), (2, 5), (1, 5), (0, 5)]
    
    get_sequential_pairs(my_list)
    
    my_dict = {(5, 2): [(5, 1)],
               (4, 1): [(5, 1)],
               (4, 0): [(3, 0), (5, 0)], 
               (3, 1): [(3, 0)]}
    
    get_flippable_coords(my_dict)
    
main()