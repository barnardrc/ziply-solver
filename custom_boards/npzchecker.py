# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:10:03 2025

@author: barna
"""
import numpy as np

dims_to_check = '7x7'

# Prints the contents to console as easy copy-paste string format
try:
    with np.load(dims_to_check + '.npz', allow_pickle=True) as data:
        # Reconstruct the set from the loaded arrays
        for arr in data.values():
            arr = np.array2string(arr, separator=', ')
            print(arr)
        
except FileNotFoundError:
    # If the file doesn't exist, start with an empty set
    print("No .npz file found")
