# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 20:14:04 2025

@author: barna
"""

import argparse

def parse_arguments():
    # ----- argparser ----- #
    
    parser = argparse.ArgumentParser(description="A solver for the Ziply directional graph puzzle.")
    
    # Display animation that simulates the path the algorithm takes
    parser.add_argument(
        '-sa','--show-animation', 
        action='store_true', 
        dest='displayAnimation',
        help="Enable the final path animation."
    )
    
    # Draw solution in browser
    parser.add_argument(
        '-ns','--no-solution', 
        action='store_false',
        dest='drawSolution',
        help="Disable drawing in the puzzle window."
    )
    
    # Print solution coords to console
    parser.add_argument(
        '-dc','--display-coords', 
        action='store_true',
        dest='displaySolutionCoords',
        help="Print the final solution coordinates to the console."
    )
    # Show heatmap of intersections
    parser.add_argument(
        '-hm','--show-heatmap', 
        action='store_true',
        dest='displayHeatmap',
        help="Show the heatmap of intersecting possible paths."
    )
    parser.add_argument(
        '-ts','--trouble-shoot', 
        action='store_true',
        dest='ts',
        help="This mode largely gives a step by step of what is occuring\
            for the OCR pipeline."
    )
    # Change simulation length
    parser.add_argument(
        '-sl', '--sim-length',
        type=int,
        default=1000,
        dest='simulationLength',
        help = "First SIMULATIONLENGTH coordinates will be simulated."
    )
    
    args = parser.parse_args()
    
    return args