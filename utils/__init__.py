# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:03:11 2025

@author: barna
"""

import platform

if platform.system() == "Windows":
    from .windows_utils import get_foreground_window
elif platform.system() == "Linux":
    from .linux_utils import get_foreground_window
elif platform.system() == "Darwin":  # macOS
    from .mac_utils import get_foreground_window
else:
    raise NotImplementedError("Foreground window detection not supported on this OS")
