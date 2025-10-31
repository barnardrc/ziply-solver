# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:10:03 2025

Run script and set dimensions you want to see. This will print them to console.

If you want to see an npz file via this script that has a checkpoint or CBS
parameter, it will have to be modified.

@author: barna
"""
import numpy as np

def get_boards(file_path = 'custom_boards/', dims_to_check = None, 
               print_to_console = None):
    
    if not dims_to_check:
        dims_to_check = input("Type dimensions to pull -> (dim1)x(dim2): ")
    
    
    # Prints the contents to console as easy copy-paste string format
    try:
        with np.load(file_path + dims_to_check + '.npz', allow_pickle=True) as data:
            
            # Reconstruct the set from the loaded arrays
            if len(data.values()) > 0:
                
                for i, arr in enumerate(data.values()):
                    arr = np.array2string(arr, separator=', ')
                    if print_to_console:
                        print(f"\nBoard {i+1}:")
                        print(arr)
                return dict(data)
            
            else:
                print("- No boards of that size exist.")
            
            
    except FileNotFoundError:
        # No file found
        print(f"{dims_to_check}.npz does not exist.")
        
if __name__ == "__main__":
    get_boards(file_path = '', print_to_console=True)