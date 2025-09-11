# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:06:58 2025

@author: barna
"""
import ctypes

class Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long)
        ]
    
    
# Returns a tuple of the foreground window ordered as:
# (top, left, width, height) for passing into MSS
def get_foreground_window():
    user32 = ctypes.windll.user32
    rect = Rect()
    hWnd = user32.GetForegroundWindow()
    winRect = user32.GetWindowRect(hWnd, ctypes.byref(rect))
    
    if winRect:
        # Return order is defined by the input order of MSS
        # for a portion of the screen to be captured
        return (rect.top, 
                rect.left, 
                rect.right - rect.left, # MSS takes width
                rect.bottom - rect.top # MSS takes height
                )
    else:
        return None