# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:10:03 2025

@author: barna
"""
import numpy as np

def main():
    dims_to_check = input("Type dimensions to pull -> (dim1)x(dim2): ")
    
    
    # Prints the contents to console as easy copy-paste string format
    try:
        with np.load(dims_to_check + '.npz', allow_pickle=True) as data:
            
            # Reconstruct the set from the loaded arrays
            if len(data.values()) > 0:
                for i, arr in enumerate(data.values()):
                    arr = np.array2string(arr, separator=', ')
                    print(f"\nBoard {i+1}:")
                    print(arr)
                    
            else:
                print("- No boards of that size exist.")
            
            
    except FileNotFoundError:
        # No file found
        print(f"{dims_to_check}.npz does not exist.")
        
if __name__ == "__main__":
    main()