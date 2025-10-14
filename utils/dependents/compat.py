# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:52:26 2025

@author: barna
"""

# compat.py
import platform
import os
import sys

def check_environment():
    system = platform.system().lower()

    # --- Windows ---
    if system == "windows":
        return "windows"

    # --- macOS ---
    elif system == "darwin":
        print("macOS detected: Ensure you granted 'Accessibility' and 'Screen Recording' permissions "
              "to this app in System Preferences → Security & Privacy.")
        return "macos"

    # --- Linux ---
    elif system == "linux":
        if "DISPLAY" in os.environ:
            print("X11 detected: Full compatibility expected.")
            return "linux-x11"
        else:
            print("No display server detected. This script requires a graphical environment.")
            sys.exit(1)

    else:
        print(f"Unsupported OS: {system}")
        sys.exit(1)
