"""Foreground-window geometry helper for X11 Linux desktops."""

import subprocess


def get_foreground_window():
    """Return ``(top, left, width, height)`` or ``None`` when unavailable."""
    try:
        window_id = subprocess.check_output(
            ["xdotool", "getactivewindow"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        raw_geometry = subprocess.check_output(
            ["xdotool", "getwindowgeometry", "--shell", window_id],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    geometry = {}
    for line in raw_geometry.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            geometry[key] = value

    try:
        return (
            int(geometry["Y"]),
            int(geometry["X"]),
            int(geometry["WIDTH"]),
            int(geometry["HEIGHT"]),
        )
    except (KeyError, ValueError):
        return None

