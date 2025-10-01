# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:31:17 2025

@author: barna
"""
import numpy as np

class Helper():
    def get_closed_coords(dict_of_coords):
        return [coord for coord_list in dict_of_coords.values() for coord in coord_list]
    
            
    def get_loc(board, target):
        coords = np.where(board == target)
        row = coords[0][0]
        col = coords[1][0]
        
        return (row, col)