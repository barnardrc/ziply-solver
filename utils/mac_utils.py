# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:03:11 2025

Utility for macOS – get the geometry of the current foreground window
in the order that MSS expects it.
    (top, left, width, height)

The values are suitable for libraries such as `mss` that expect a
top‑left origin (origin = 0,0 at the *top* of the screen).

@author: barna

"""

from __future__ import annotations
import sys
from typing import Tuple, Optional

# Double check platform
if sys.platform != "darwin":
    raise ImportError("mac_window.py can only be imported on macOS (darwin)")

# Imports from the PyObjC bridge
import Quartz               # type: ignore   (Quartz is provided by pyobjc)
from AppKit import NSScreen  # type: ignore


def _get_main_display_height() -> int:
    """
    Returns the height (in points) of the primary display.
    macOS uses a bottom‑left origin for most Quartz coordinates,
    while `mss` expects a top‑left origin.  Knowing the screen height
    lets us flip the Y axis.
    """
    # ``NSScreen.mainScreen`` can be ``None`` (e.g. when no UI session)
    main_screen = NSScreen.mainScreen()
    if main_screen is None:
        # Fallback – assume the common 1080p height
        return 1080
    # Frame is in “points”, not pixels.  For screenshot purposes that is fine.
    frame = main_screen.frame()
    return int(frame.size.height)


def get_foreground_window() -> Optional[Tuple[int, int, int, int]]:
    """
    Returns a tuple (top, left, width, height) describing the currently
    active (foreground) window on macOS.

    If something goes wrong (e.g. no window could be identified), ``None`` is
    returned – matching the behaviour of the Windows helper.
    """
    
    # Determine foreground window
    workspace = Quartz.NSWorkspace.sharedWorkspace()
    front_app = workspace.frontmostApplication()
    if front_app is None:
        return None

    front_pid = front_app.processIdentifier()

    # Filter unimportant windows
    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListOptionIncludingWindow
    )
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

    # Locate target window
    target_window = None
    for win in window_list:
        # `kCGWindowOwnerPID` is the process ID, `kCGWindowLayer` is the Z‑order layer.
        if (
            win.get("kCGWindowOwnerPID") == front_pid
            and win.get("kCGWindowLayer") == 0
            and not win.get("kCGWindowIsOnscreen") is False
        ):
            # Some windows (e.g. those belonging to a fullscreen game) may not
            # report a bounds dictionary – skip them.
            if "kCGWindowBounds" in win:
                target_window = win
                break

    if target_window is None:
        return None


    # Extract the geometry.  The dictionary looks like:
    # {'X': 0, 'Y': 23, 'Width': 1440, 'Height': 877}
    bounds = target_window["kCGWindowBounds"]
    left   = int(bounds["X"])
    bottom = int(bounds["Y"])                 # Bottom‑left origin
    width  = int(bounds["Width"])
    height = int(bounds["Height"])

    #Convert to *top‑left* coordinates required by `mss`
    screen_height = _get_main_display_height()
    top = screen_height - (bottom + height)

    return (top, left, width, height)



# Direct exectution check
if __name__ == "__main__":
    geom = get_foreground_window()
    if geom:
        print(f"Foreground window geometry (top, left, w, h): {geom}")
    else:
        print("Unable to locate a foreground window.")

