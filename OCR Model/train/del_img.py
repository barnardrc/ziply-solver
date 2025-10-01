# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 09:36:57 2025

@author: barna
"""

import os
import glob

def del_img(directory_path):
    # List to store the paths of files found
    files_to_delete = []
    
    # 1. Path Check
    if not os.path.isdir(directory_path):
        print("No path")
        return
    
    all_exts = ['-0.png', '-1.png', '-2.png', '-3.png', '-4.png', '-5.png', '-6.png',
                 '-7.png', '-8.png', '-9.png', '-10.png', '-11.png', '-12.png', 
                 '-13.png', '-14.png', '-15.png', '-16.png', '-17.png']

    to_exclude = ['-0.png', '-2.png', '-3.png', '-7.png', '-9.png', '-10.png',
                  '-13.png', '-14.png']
    
    to_delete = [ext for ext in all_exts if ext not in to_exclude]
    
    
    
    for ext in to_delete:
        search_pattern = os.path.join(directory_path, f"*{ext}")
        files_to_delete.extend(glob.glob(search_pattern, recursive = False))
        print(search_pattern)
    
    if not files_to_delete:
        return [] # Return empty list if none are found

    # 3. File Deletion Logic (NEW CODE)
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path) # <-- THIS IS THE DELETION STEP
            print(f"Successfully deleted: {file_path}")
            deleted_count += 1
        except OSError as e:
            # Handles permission errors, locked files, etc.
            print(f"Error deleting {file_path}: {e}")

    print(f"\nCompleted deletion. Total deleted: {deleted_count}")
    # Return the original list of files that were targeted
    return files_to_delete
    
def main():
    file_path = r'C:\Users\barna\Documents\ziply-solver\OCR Model\train'
    x = del_img(file_path)
    

main()