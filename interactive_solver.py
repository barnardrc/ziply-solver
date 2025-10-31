# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:24:25 2025

This script essentially builds the ziply game board locally with passed arrays,
or generates a random board with specific inputs 
- Dimensions
- number of checkpoints

Set params in main() at the bottom of script

@author: barna
"""

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageColor, ImageFont # Import Pillow
import random
from custom_boards.npzchecker import get_boards

# The game board class, responsible for tracking and communicating
# the state of the game board
class Board():
    _instance = None
    
    # Singleton
    def __new__(cls, board_array, cell_size=100, show_costs = False):
        if cls._instance is None:
            # Create a new instance if one doesn't exist
            cls._instance = super(Board, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, board = None, cell_size = 100, show_costs = False):
        if not hasattr(self, '_initialized'):
            self.show_costs = show_costs
            
            # Init board: a 2D numpy array
            self.board = board
            self.dim = board.shape
            self.cell_size = cell_size
            self.cost_arr = None
            self.g_cost_arr = None
            self.f_cost_arr = None
            self.open_list = []
            self.closed_list = []
            
            
            # Tuning params
            self.h_cost_tuner = 1
            self.g_cost_tuner = 0.45
            
            # Checkpoint handling
            self.current_checkpoint = 1
            self.current_checkpoint_coords = None
            self.num_checkpoints = self.get_num_checkpoints()
            self.update_current_checkpoint_coords()
            self.checkpoints = None
            self.get_list_of_checkpoints()
            
            # History handling
            self.last_row = None
            self.last_col = None
            self.path = []
            
            # Solved state (not implemented)
            self.solved = False
        
    # Updates checkpoint, the respective coordinates, and calls to update
    # the intersection costs
    def increase_checkpoint(self):
        if not self.solved:
            self.current_checkpoint += 1
            self.update_current_checkpoint_coords()
        self.update_h_cost()
        self.update_g_cost_arr()
        
    # Updates checkpoint, the respective coordinates, and calls to update
    # the intersection costs
    def decrease_checkpoint(self):
        self.current_checkpoint -= 1
        self.update_current_checkpoint_coords()
        self.update_h_cost()
        self.update_g_cost_arr()
        
    def get_list_of_checkpoints(self):
        row, col = np.where(self.board > 0)
        self.checkpoints = list(zip(row, col))
    
    # Checks if coord value is in the list of checkpoints
    def in_checkpoints(self, value: tuple) -> bool:
        return value in self.checkpoints
    
    # i is an adjustment for dragging back off of a checkpoint. If a checkpoint
    # is drug off of, clearing the space, the game needs to know which 
    # checkpoint to check against for decreasing it. Since current_checkpoint
    # increases immediately upon entering a checkpoint space, when moving back
    # off of it, is_checkpoint needs to check against the current - 1 point
    def is_checkpoint(self, value: tuple, i: int = 0) -> bool:
        row, col = value
        #print(row, col)
        return self.board[row, col] == self.current_checkpoint - i
        
    def is_solved(self):
        return self.solved
        
    def update_last_coords(self, row, col):
        self.last_row = row
        self.last_col = col
        
    def update_current_checkpoint_coords(self):
        if self.current_checkpoint <= self.num_checkpoints:
            row, col = np.where(self.board == self.current_checkpoint)
            coords = list(zip(row, col))
        
            # Pulls tuple out of list
            self.current_checkpoint_coords = coords[0]
    
    def get_current_checkpoint_coords(self):
        if self.current_checkpoint <= self.num_checkpoints:
            return self.current_checkpoint_coords
        
    def is_adjacent(self, row, col):
        last_path_row = self.path[-1][0]
        last_path_col = self.path[-1][1]
        is_adjacent = (abs(row - last_path_row) == 1 and col == last_path_col) or \
                         (abs(col - last_path_col) == 1 and row == last_path_row)
        
        return is_adjacent
    
    # Gets the total amount of checkpoints
    def get_num_checkpoints(self):
        return np.sum(self.board > 0)
    
    # Reset the cost array
    def _reset_cost_arr(self):
        H = self.dim[0]
        W = self.dim[1]
        self.cost_arr = np.zeros((H, W))
        
    # Gets the respective coordinates of a value's location for two values
    def _get_coords(self, p1: int, p2: int) -> tuple:
        
        if self.board.ndim < 2:
            raise ValueError("Expected atleast 2 dimensions.")
        
        p1_coords = np.where(self.board == p1)
        p2_coords = np.where(self.board == p2)
        
        # Unpacking array return from np.where, slicing to get the first
        # 2 dims of location
        p1_coords = list(zip(*p1_coords[:2]))
        p2_coords = list(zip(*p2_coords[:2]))
        
        # Isolating coordinate tuples from lists before return
        return p1_coords[0], p2_coords[0]
    
    # Gets each coordinate within a bounding box between two points
    def _get_bounding_box(self, p1, p2):
        bbox = []
        
        p1, p2 = self._get_coords(p1, p2)

        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        
        start_x = min(x1, x2)
        end_x = max(x1, x2)
        start_y = min(y1, y2)
        end_y = max(y1, y2)

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                coords = (x, y)
                bbox.append(coords)

        
        return bbox
    
    # Increment values in array for each occurence of a coordinate
    def _increment_cost(self, coords: list):
        for coord in coords:
            self.cost_arr[coord[0], coord[1]] += 1
            
    # Calculates the cost of a cell based on how many bounding boxes
    # intersect there.
    def update_intersection_costs(self):
        # Reset the cost array each time its recalculated
        self._reset_cost_arr()

        # current_checkpoint is decreased by 1 here, because it represents
        # the current target, and the intersection doesn't need to update
        # until that target has been reached.
        for i in range(self.current_checkpoint-1, self.num_checkpoints):
            
            bbox = self._get_bounding_box(i, i+1)
            self._increment_cost(bbox)
        
    def manhattan_dist(self, p1, p2):
        
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        
        return abs(y2 - y1) + abs(x2 - x1)
    
                
    def update_h_cost(self):
        self.update_intersection_costs()
        self.cost_arr = np.around((self.cost_arr *
                         self.h_cost_tuner
                         ), decimals=2)
        
    
    def update_g_cost_arr(self):
        H = self.dim[0]
        W = self.dim[1]
        
        # reset array each time
        self.g_cost_arr = np.zeros((H, W))
        
        for coord, _ in np.ndenumerate(self.board):
            self.g_cost_arr[coord[0], coord[1]] =np.around((
                self.manhattan_dist(self.current_checkpoint_coords, coord) * 
                self.g_cost_tuner), decimals=2)

        #print(self.g_cost_arr)
    
    def update_f_cost_arr(self):
        H = self.dim[0]
        W = self.dim[1]
        
        self.f_cost_arr = np.zeros((H, W), dtype=int)
        
        self.f_cost_arr = np.around(self.g_cost_arr + self.cost_arr, decimals=2)
    
class Draw:
    def __init__(self, board, canvas):
        # Global variables for the transparent layer and its Tkinter PhotoImage
        self.transparent_image_pil = None
        self.transparent_image_tk = None
        self.transparent_image_id = None
        self.animation_counter = 0
        self.circle_id_map = {}
        
        # For static board image
        self.static_image_pil = None
        self.static_image_tk = None
        self.static_image_id = None
        
        # For cost image
        self.cost_image_pil = None
        self.cost_image_tk = None
        self.cost_image_id = None
        
        # For g cost image
        self.g_cost_image_pil = None
        self.g_cost_image_tk = None
        self.g_cost_image_id = None
        
        # For open list image
        # For g cost image
        self.open_pil = None
        self.open_tk = None
        self.open_id = None
        
        # Canvas
        self.canvas = canvas
        
        # Board
        self.board = board
        
        self.show_costs = board.show_costs
        
    # Initial drawing of the board (grid, circles, numbers)
    def draw_board(self):
        H, W = self.board.dim
        border_width = 3
        grid_line_width = 2.3
        x_end = W * self.board.cell_size  # Total canvas width
        y_end = H * self.board.cell_size  # Total canvas height
        
        
        
        # Draw grid lines with Tkinter

        
        
        # interior grid lines
        for i in range(H):
            self.canvas.create_line(0, i * self.board.cell_size, W * self.board.cell_size, i * self.board.cell_size,
                               fill="#cccacf", width=grid_line_width, tags="grid_lines")
        for j in range(W):
            self.canvas.create_line(j * self.board.cell_size, 0, j * self.board.cell_size, H * self.board.cell_size,
                               fill="#cccacf", width=grid_line_width, tags="grid_lines")
        
        # exterior grid lines
        self.canvas.create_line(0, border_width, x_end, border_width, fill = '#000000', width = border_width, tag = 'grid_lines')
        self.canvas.create_line(0, y_end, x_end, y_end, fill = '#000000', width = 3, tag = 'grid_lines')
        self.canvas.create_line(border_width, 0, border_width, y_end, fill = '#000000', width = 3, tag = 'grid_lines')
        self.canvas.create_line(x_end, 0, x_end, y_end, fill = '#000000', width = 3, tag = 'grid_lines')
        
    def draw_static_board_elements(self, scale_factor=2, circle_size=20):
    
        H, W = self.board.dim
        canvas_width = W * self.board.cell_size
        canvas_height = H * self.board.cell_size
        
        # Create a larger, temporary PIL image for supersampling
        self.temp_pil_image = Image.new("RGBA", (canvas_width * scale_factor, canvas_height * scale_factor), (0, 0, 0, 0))
        draw_pil = ImageDraw.Draw(self.temp_pil_image)
        
        # Draw circles and numbers with Pillow
        for r in range(H):
            for c in range(W):
                if self.board.board[r, c] != 0:
                    # Calculate coordinates and dimensions with scaling
                    x0 = (c * self.board.cell_size + circle_size) * scale_factor
                    y0 = (r * self.board.cell_size + circle_size) * scale_factor
                    x1 = ((c + 1) * self.board.cell_size - circle_size) * scale_factor
                    y1 = ((r + 1) * self.board.cell_size - circle_size) * scale_factor
                    
                    # Draw the circle and store its ID (optional, but good for management)
                    draw_pil.ellipse((x0, y0, x1, y1), fill="#202021", outline="#9370DB", width=4 * scale_factor)
                    
                    # Draw the number on the circle
                    text_x = (c * self.board.cell_size + self.board.cell_size / 2) * scale_factor
                    text_y = (r * self.board.cell_size + self.board.cell_size / 2) * scale_factor
                    font_size = int(self.board.cell_size / 4 * scale_factor)
                    font = ImageFont.truetype("arial.ttf", font_size) # Make sure you have arial.ttf
                    
                    draw_pil.text((text_x, text_y), str(self.board.board[r, c]), fill="white", font=font, anchor="mm")
                    
        # Downscale the temporary image
        self.static_image_pil = self.temp_pil_image.resize((canvas_width, canvas_height), Image.LANCZOS)
        self.static_image_tk = ImageTk.PhotoImage(self.static_image_pil)
        
        # Place the static image on the canvas
        self.static_image_id = self.canvas.create_image(0, 0, image=self.static_image_tk, anchor="nw", tags="static_layer")
        self.canvas.lower(self.static_image_id)
        
    def get_rgba(self, color_name, alpha = 255):
        rgb_tuple = ImageColor.getrgb(color_name)
        #print(f"{color_name}: {rgb_tuple}: {alpha}")
        return rgb_tuple + (int(alpha),)
    
    # --- Drawing Functions ---
    def create_transparent_layer(self):
        H, W = self.board.dim
        canvas_width = W * self.board.cell_size
        canvas_height = H * self.board.cell_size

        # Create a new transparent PIL image
        self.transparent_image_pil = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        self.transparent_image_tk = ImageTk.PhotoImage(self.transparent_image_pil)

        # Place the image on the canvas. It should be at (0,0) and anchor 'nw' (north-west).
        # We remove the old image if it exists to refresh it.
        if self.transparent_image_id:
            self.canvas.delete(self.transparent_image_id)
            
        self.transparent_image_id = self.canvas.create_image(0, 0, image=self.transparent_image_tk, anchor="nw", tags="transparent_layer")
        self.canvas.lower(self.transparent_image_id, "grid_lines") # Ensure transparent layer is below circles
    
    #responsible for drawing the cost numbers
    def draw_hcost_layer(self):
        # responsible for drawing the h cost numbers and f cost numbers
        H, W = self.board.dim
        canvas_width = W * self.board.cell_size
        canvas_height = H * self.board.cell_size
        
        # h cost font
        cost_font_size = int(self.board.cell_size / 7)
        cost_font = ImageFont.truetype("arial.ttf", cost_font_size)
        
        # f cost font
        f_cost_font_size = int(self.board.cell_size / 6)
        f_cost_font = ImageFont.truetype("arial.ttf", f_cost_font_size)
        
        temp_pil_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw_pil = ImageDraw.Draw(temp_pil_image)
        
        for r in range(H):
            for c in range(W):
                # H costs
                cost_value = self.board.cost_arr[r, c]
                text_x = (c + 1) * self.board.cell_size - 5
                text_y = r * self.board.cell_size + 5
                draw_pil.text((text_x, text_y), str(cost_value), fill="blue", 
                              font=cost_font, anchor="rt")
        
        for r in range(H):
            for c in range(W):
                # f costs
                f_cost_value = self.board.f_cost_arr[r, c]
                text_to_draw = str(f_cost_value)
                # No ct align :(
                text_width = draw_pil.textlength(text_to_draw, font=f_cost_font)
                text_x = (c * self.board.cell_size + self.board.cell_size / 2) - (text_width / 2)
                text_y = r * self.board.cell_size
                draw_pil.text((text_x, text_y), str(f_cost_value), fill="black", 
                              font=f_cost_font)
        
        # Update the canvas
        self.cost_image_pil = temp_pil_image
        self.cost_image_tk = ImageTk.PhotoImage(self.cost_image_pil)
    
        if self.cost_image_id is None:
            self.cost_image_id = self.canvas.create_image(0, 0, image=self.cost_image_tk, anchor="nw", tags="cost_layer")
            self.canvas.lower(self.cost_image_id, "grid_lines")
        else:
            self.canvas.itemconfig(self.cost_image_id, image=self.cost_image_tk)
    
    #responsible for drawing the g cost numbers
    def draw_gcost_layer(self):
        # responsible for drawing the g cost numbers
        H, W = self.board.dim
        canvas_width = W * self.board.cell_size
        canvas_height = H * self.board.cell_size
        
        cost_font_size = int(self.board.cell_size / 7)
        cost_font = ImageFont.truetype("arial.ttf", cost_font_size)
        
        temp_pil_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw_pil = ImageDraw.Draw(temp_pil_image)
        
        for r in range(H):
            for c in range(W):
                # G costs
                g_cost_value = self.board.g_cost_arr[r, c]
                text_gx = c * self.board.cell_size + 5
                text_gy = r * self.board.cell_size + 5
                draw_pil.text((text_gx, text_gy), str(g_cost_value), 
                              fill="green", font=cost_font, anchor="lt")
                
        # Update the canvas
        self.g_cost_image_pil = temp_pil_image
        self.g_cost_image_tk = ImageTk.PhotoImage(self.g_cost_image_pil)
    
        if self.g_cost_image_id is None:
            self.g_cost_image_id = self.canvas.create_image(0, 0, image=self.g_cost_image_tk, anchor='nw', tags="gcost_layer")
            self.canvas.lower(self.g_cost_image_id, "grid_lines")
        else:
            self.canvas.itemconfig(self.g_cost_image_id, image=self.g_cost_image_tk)
            
    def draw_path_layer(self, scale_factor = 2):
        
        line_color_rgba = self.get_rgba('mediumpurple', 255)
        open_square_color_rgba = (155, 60, 90, 80)
        closed_square_color_rgba = (194, 85, 190, 100)
        
        if self.transparent_image_pil is None:
            self.create_transparent_layer()

        H, W = self.board.dim
        canvas_width = W * self.board.cell_size
        canvas_height = H * self.board.cell_size
        
        # Clear the previous drawings on the PIL image
        temp_pil_image = Image.new("RGBA", (canvas_width * scale_factor, canvas_height * scale_factor), (0, 0, 0, 0))
        draw_pil = ImageDraw.Draw(temp_pil_image)
        
        # Draw the transparent squares for each cell in the path
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Draw squares in the open list
        """
        for row, col in self.board.path:
            for direc in directions:
                new_row = row + direc[0]
                new_col = col + direc[1]
                x0 = new_col * self.board.cell_size * scale_factor
                y0 = new_row * self.board.cell_size * scale_factor
                x1 = (new_col + 1) * self.board.cell_size * scale_factor
                y1 = (new_row + 1) * self.board.cell_size * scale_factor
                
                draw_pil.rectangle([x0, y0, x1, y1], fill=open_square_color_rgba)
        """
        
        # Draw squares in the closed list
        for row, col in self.board.path:
                x0 = col * self.board.cell_size * scale_factor
                y0 = row * self.board.cell_size * scale_factor
                x1 = (col + 1) * self.board.cell_size * scale_factor
                y1 = (row + 1) * self.board.cell_size * scale_factor
                
                draw_pil.rectangle([x0, y0, x1, y1], fill=closed_square_color_rgba)
            
        # Draw the continuous line over the squares 
        if len(self.board.path) > 1:
            # Create a list of pixel coordinates from the board.path's cell coordinates
            pixel_path = []
            for row, col in self.board.path:
                x_center = (col * self.board.cell_size + self.board.cell_size / 2) * scale_factor
                y_center = (row * self.board.cell_size + self.board.cell_size / 2) * scale_factor
                pixel_path.append((x_center, y_center))

            # Draw the line segments with a specified width
            draw_pil.line(pixel_path, fill=line_color_rgba, width=40 * scale_factor, joint="curve")
            
            # Draw a circle at each point in the path
            point_radius = 19.25 * scale_factor
            for x_center, y_center in pixel_path:
                x0 = x_center - point_radius
                y0 = y_center - point_radius
                x1 = x_center + point_radius
                y1 = y_center + point_radius
                draw_pil.ellipse((x0, y0, x1, y1), fill=line_color_rgba) # Use ellipse to draw the circle


                
                
        # Update the Tkinter PhotoImage from the modified PIL image
        self.transparent_image_pil = temp_pil_image.resize((canvas_width, canvas_height), Image.LANCZOS)
        self.transparent_image_tk = ImageTk.PhotoImage(self.transparent_image_pil)
        self.canvas.itemconfig(self.transparent_image_id, image=self.transparent_image_tk)
        self.canvas.lower("grid_lines")
        
        # Update the circle outlines based on the path
        for coords, circle_id in self.circle_id_map.items():
            if coords in self.board.path:
                self.canvas.itemconfig(circle_id, outline="#785abe", width=5) # Example for a visited circle
            else:
                self.canvas.itemconfig(circle_id, outline="#202021", width=1) # Reset to default
               
        self.canvas.lift("static_layer")
        
    def animate_click_feedback(self, row, col, frame=0, circle_id = None):
        max_frames = 20
        starting_radius = 25  # Start with a small radius
        max_radius = self.board.cell_size / 2 - starting_radius
        
        # Calculate radius
        radius = starting_radius + (frame / max_frames) * max_radius
        
        # Color interpolation for fade (sick). Tk does not natively support
        # alpha channels. So here, we are interpolating from the target color to
        # the background color, then eventually deleting the circle
        start_rgb = ImageColor.getrgb('mediumpurple')
        end_rgb = ImageColor.getrgb('mintcream')
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (frame / max_frames))
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (frame / max_frames))
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (frame / max_frames))
        fill_color = f'#{r:02x}{g:02x}{b:02x}'
        
        # Calculate screen coordinates
        x_center = col * self.board.cell_size + self.board.cell_size / 2
        y_center = row * self.board.cell_size + self.board.cell_size / 2
        x0 = x_center - radius
        y0 = y_center - radius
        x1 = x_center + radius
        y1 = y_center + radius
        
        
        if circle_id is None:
            # Create a new oval item on the canvas for the first frame
            circle_id = self.canvas.create_oval(x0, y0, x1, y1, 
                                           fill=fill_color, 
                                           outline="", width=0, tag = 'fading_circle')
            # Note: Tkinter can't handle RGBA directly for fill, so we can simulate it with a solid fill and use 'after' to delete it. For real transparency, you'd need a more complex solution or to stick with PIL.
            # A simpler approach for the animation is to just fade the size.
            self.canvas.lower('fading_circle')
            
        else:
            # Update the existing circle's size
            self.canvas.coords(circle_id, x0, y0, x1, y1)
            self.canvas.itemconfig(circle_id, fill=fill_color)
        
        if frame < max_frames:
            # Schedule the next frame
            self.canvas.after(25, self.animate_click_feedback, row, col, frame + 1, circle_id)
        else:
            # Delete the circle when the animation is done
            self.canvas.delete(circle_id)
        
    def on_drag_start(self, event):
        col = event.x // self.board.cell_size
        row = event.y // self.board.cell_size
        
        # Start the visual feedback animation for the click
        self.animation_counter += 1
        self.animate_click_feedback(row, col)
        
        # On first time dragging.
        if len(self.board.path) == 0:
            start_point = self.board.get_current_checkpoint_coords()
            # If you click outside the board, do nothing
            if not (0 <= row < self.board.dim[0] and 0 <= col < self.board.dim[1]):
                return
            # Only draw on starting a drag if its a new point
            if len(self.board.path) == 0 and (row, col) == start_point:
                self.board.path.append((row, col))
                self.board.increase_checkpoint()
                self.board.closed_list.append((row, col))
                if self.show_costs:
                    # Update costs
                    self.board.update_h_cost()
                    self.board.update_f_cost_arr()
                    
                    # Draw cost layers
                    self.draw_hcost_layer()
                    self.draw_gcost_layer()
                
            # Store the initial cell
            self.board.update_last_coords(row, col)
        
        # On clicking a visited checkpoint, reset the path to that point.
        elif (self.board.in_checkpoints((row, col)) and 
              (row, col) in self.board.path
              ):
            # Get clicked checkpoint and index of loc in path
            cp = self.board.board[row, col]
            idx = self.board.path.index((row, col))
            
            # pop all path after current loc
            self.board.path[idx + 1:] = []   
            # Reset checkpoint
            self.board.current_checkpoint = cp + 1
            self.board.update_current_checkpoint_coords()
            self.draw_path_layer()
            self.board.update_last_coords(row, col)
            
        else:
            # Store the initial cell
            self.board.update_last_coords(row, col)
            
            return
        
    def on_drag_motion(self, event):
        col = event.x // self.board.cell_size
        row = event.y // self.board.cell_size
        
        if self.board.last_row is None or self.board.last_col is None or not self.board.path:
            return
        
        # Do nothing if draging outside of the board
        if not (0 <= row < self.board.dim[0] and 0 <= col < self.board.dim[1]):
            return
        
        # Check if dragging into new grid space
        if row != self.board.last_row or col != self.board.last_col:
            
            if self.show_costs:
                # Update cost layers
                self.board.update_h_cost()
                self.board.update_f_cost_arr()
                # G cost only updated on new checkpoint
                
                # Draw cost layers
                self.draw_hcost_layer()

            
            # The is_adjacent check needs to check against the last drawn path coord
            adjacent = self.board.is_adjacent(row, col)
            
            # First check if dragging backwards.
            if len(self.board.path) >= 2 and (row, col) == self.board.path[-2]:
                
                # Check if moved off of checkpoint
                if self.board.is_checkpoint((self.board.last_row, self.board.last_col), 1):
                    self.board.decrease_checkpoint()
                    if self.show_costs:
                        self.draw_gcost_layer()
                        self.board.update_f_cost_arr()
                        
                        # Draw cost layers
                        self.draw_hcost_layer()
                
                
                self.board.path.pop()
                self.draw_path_layer()
                
                # Update last_row, last_col
                self.board.update_last_coords(self.board.path[-1][0], self.board.path[-1][1])
                


            
            # Now check if it is a valid move space
            elif adjacent and (row, col) not in self.board.path and self.board.current_checkpoint <= self.board.num_checkpoints:
                
                # Then Check if it is a checkpoint space
                if (self.board.in_checkpoints((row, col)) and 
                    not self.board.is_checkpoint((row, col))
                    ):
                    return
                
                # Moved onto checkpoint
                elif (self.board.in_checkpoints((row, col)) and 
                      self.board.is_checkpoint((row, col))
                    ):
                    self.board.path.append((row, col))
                    self.draw_path_layer() # Redraw the path with the new segment
                    self.board.update_last_coords(row, col)
                    self.board.increase_checkpoint()
                    
                    if self.show_costs:
                        # Update g cost
                        self.draw_gcost_layer()
                        self.board.update_f_cost_arr()
                        self.draw_hcost_layer()
                    
                else:
                    self.board.path.append((row, col))
                    self.draw_path_layer() # Redraw the path with the new segment
                    self.board.update_last_coords(row, col)

                    
    def on_drag_release(self, event):
        # Reset last_row/col so that the next drag starts fresh
        self.board.update_last_coords(None, None)
        pass    
    
# Generates a NumPy array with random checkpoints.
# This is more of a proof of concept, as there is no guarantee that the board
# generated will be solvable.
def generate_random_board(height: int, width: int, num_checkpoints: int):
    
    if num_checkpoints > height * width:
        raise ValueError("Number of checkpoints cannot exceed board size.")
    
    # Create an empty board
    board = np.zeros((height, width), dtype=int)
    
    # Create a list of all possible coordinates
    all_coords = [(r, c) for r in range(height) for c in range(width)]
    
    # Shuffle the coordinates to get random positions
    random.shuffle(all_coords)
    
    # Place the checkpoints
    checkpoint_coords = all_coords[:num_checkpoints]
    for i, (r, c) in enumerate(checkpoint_coords):
        board[r, c] = i + 1
        
    return board


def main():
    ### If you want to see a specific, passed array, skip these parameters.
    # this is the default behavior. Set your array in the else statement. ###
    
    # If not using a random board or a specific passed array, use this
    # 1. Set dimensions for npz file to check
    all_boards = get_boards(dims_to_check = '6x6')
    # 2. Set generated boards to True
    generated_boards = False
    # 3. Optionally, if you know the key of the board you want to see
    # set it here.
    desired_board = None
    
    # Optionally, generate a random board to display. The array string will print to
    # console.
    random_board = False
    
    if random_board:
        height = 6
        width = 6
        num_checkpoints = 3
    
        board = generate_random_board(height, width, num_checkpoints)
        title = 'Interactive Board'
    
    #
    elif generated_boards:
        keys = list(all_boards.keys())
        if desired_board is not None:
            board = all_boards[desired_board]
            title = f'{desired_board}'
            
        else:
            random_key = random.choice(keys)
            board = all_boards[random_key]
            title = f'{random_key}'
        
    else:
        board = np.array(
[[0, 0, 0, 0, 0, 1],
 [7, 0, 0, 0, 8, 0],
 [5, 6, 0, 2, 0, 0],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 3],
 [0, 4, 0, 0, 0, 0]]
        )
        
        title = 'Interactive Board'
    
    board = Board(board, show_costs = False, cell_size = 100)
    print(np.array2string(board.board, separator= ', '))
    
    # Setup Tkinter Window
    root = tk.Tk()
    root.title(title)
    
    H, W = board.dim
    canvas = tk.Canvas(root, width=W * board.cell_size, height=H * board.cell_size, 
                       bg="mint cream")
    canvas.pack(padx= 200, pady = 200)
    view_controller = Draw(board, canvas)
    
    view_controller.draw_board()
    view_controller.draw_static_board_elements()
    view_controller.create_transparent_layer() # Initialize the transparent layer after canvas is set up
    
    # Bind the drag events
    view_controller.canvas.bind("<Button-1>", view_controller.on_drag_start)
    view_controller.canvas.bind("<B1-Motion>", view_controller.on_drag_motion)
    view_controller.canvas.bind("<ButtonRelease-1>", view_controller.on_drag_release)
    
    
    root.mainloop()
    
if __name__ == "__main__":
    main()