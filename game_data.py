# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 18:28:45 2025

@author: barna
"""

import mss
import cv2 as cv
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from collections import Counter
from utils import get_foreground_window
from tensorflow import keras

class GameData:
    def __init__(self):
        # debug mode
        self.ts = False
        
        # Predicion model
        self.model = keras.models.load_model("mnist_custom_digits.keras")
        
        # Binary Threshold
        self.bin_thresh = 150
        
        # Canny Edge Detector Paramaters
        self.max_lowThreshold = 100
        self.ratio = 3
        self.kernel_size = 3
        self.cannyThreshold = 60
        
        # Padding in pixels to define the Region of Interest
        self.roiThreshold = 160
        
        # These edges are that of the circles that are used to determine
        # the region of interest, not detected game edges
        self.topEdge = None
        self.botEdge = None
        self.leftEdge = None
        self.rightEdge = None
        self.xOffsetRel = None
        self.yOffsetRel = None
        self.xOffsetAbs = None
        self.xOffsetAbs = None
        
        self.pieces = None
        
        self.grid_locations = None
        self.dimensions = None
        self.image = None
        self.circles = None
        self.orderedCircles = None
        self.squares = None
        # Edges of the game board
        self.edges = None
        self.ocr_captures = None
        self.predictions = None
        self.xFactor = None
        self.yFactor = None
        self.pixel_coords = None
        
    def toggle_ts_mode(self):
        self.ts = not self.ts
        if self.ts:
            print("\n---------------DEBUG MODE--------------\n")
        return self
        
    def get_window_rect(self):
        self.dimensions = get_foreground_window()
        
        return self
    
    
    # Takes win dimension and captures img, storing as array in self.image
    def window_capture(self) -> "GameData":
        if len(self.dimensions) != 4:
            raise ValueError(f"Expected 4 dimensions (top, left, width, height), got {len(self.dimensions)}")
            
        with mss.mss() as sct:
            monitor = {
                "top" : self.dimensions[0],
                "left" : self.dimensions[1],
                "width" : self.dimensions[2],
                "height" : self.dimensions[3]
                }
            
            # Take image
            sct_img = sct.grab(monitor)
            cv.waitKey(0)
            
            # Save image in memory as array
            self.image = np.array(sct_img)[:, :, :3]  # drop alpha channel if present
            
            if self.ts:
                print(f"dimensions returned by get_window_rect: \n{self.dimensions}")
                cv.imshow("Initial Window Capture", self.image)
                cv.waitKey(0)
                
            return self
        
    # Takes the initial img and finds the most common cluster of circles
    def detect_circles(self):
        gray = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        
        gray = cv.medianBlur(gray, 5)
        
        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp = 1, 
            minDist = 20, # Min distance between circle centers
            param1 = 50, # Upper threshhold for Canny edge detector
            param2 = 50, # Accumulator threshhold
            minRadius = 15, # Smallest circle radius
            maxRadius = 30 # Largest circle radius
            )
        
        self.circles = np.uint16(np.around(circles))
        
        if circles is not None and self.ts:
            circles = np.uint16(np.around(circles))
            print(f"\ncircle coordinates: \n{circles}")
            image = np.ascontiguousarray(np.array(self.image)[:, :, :3], dtype=np.uint8)
            for (x, y, r) in circles[0, :]:
                
                # Draw circle outline
                cv.circle(image, (x, y), r, (0, 255, 0), 2)
            
            cv.imshow("Circles Detected by detect_circles", image)
            cv.waitKey(0)
        
        return self

    
    # ------ Helper Funcs for detect_clusters ------ # below        ###
    #takes a list and returns the most common items                 #|#
    def _get_most_common(self, labels: list, n: int):               #|#
        counts = Counter(labels)                                    #|#
        main_labels = counts.most_common(n)[0][0]                   #|#
        return main_labels                                          #|#
                                                                    #|#
    # Perform the actual clustering of the circle array             #|#
    def _filter_circles(self, circleArray, labels, main_label):     #|#
        filteredCircles = []                                        #|#
                                                                    #|#
        for i, (x, y, r) in enumerate(circleArray[0, :]):           #|#
            if labels[i] == main_label:                             #|#
                filteredCircles.append((x, y, r))                   #|#
                                                                    #|#
        return np.array(filteredCircles)                            #|#

    # Cluster detection to further isoalte the circles within the game prior to
    # edge detection
    def detect_clusters(self):
        #print(f"Before detect_clusters: {self.circles}")
        # Isolating x and y coords within the return from HoughCircles
        points = np.array([[x, y] for x, y, r in self.circles[0, :]])
        
        
        Z = linkage(points, method = 'average')

        max_distance = 500
        labels = fcluster(Z, max_distance, criterion = 'distance')
        labelsFilter = self._get_most_common(labels, 2)
        
        self.circles = self._filter_circles(self.circles, labels, labelsFilter)
        
        #print(f"Circles in GameData:\n{self.circles}")
        if self.ts:
            print(f"\nlabels for Clustering: {labels}")
        
        return self
    
    # Get circle relative boundary to identify a region of interest, then increase
    # it by some threshold for follow on edge detection.
    def get_circle_region_bounds(self):
        x_coords = [c[0] for c in self.circles]
        y_coords = [c[1] for c in self.circles]
        
        #print(f"x_coords for circles: \n{x_coords}")
        #print(f"y_coords for circles: \n{y_coords}")
        print()
        
        minx = min(x_coords)
        maxx = max(x_coords)
        miny = min(y_coords)
        maxy = max(y_coords)
        
        self.topEdge = miny - self.roiThreshold
        self.botEdge = maxy + self.roiThreshold
        self.leftEdge = minx - self.roiThreshold
        self.rightEdge = maxx + self.roiThreshold
        
        width = self.rightEdge - self.leftEdge
        height = self.botEdge - self.topEdge
        
        if self.ts:
            print(f"dimensions before CRB: \n{self.dimensions}")
        
        # Calculating absolute placeholders for action functions
        self.yOffsetAbs = self.dimensions[0]
        self.xOffsetAbs = self.dimensions[1]

        
        self.dimensions = tuple(map(int, (self.topEdge, 
                                          self.leftEdge, 
                                          width, height
                                          )
                                    )
                                )
        

        if self.ts:
            print()
            print(f"Dimensions returned by crb: \n{self.dimensions}")
            
        return self
    
    # Takes a slice of self.image and stores it to self.image
    def get_roi(self):
        top, left, width, height = self.dimensions
        top, left, width, height = map(int, (top, left, width, height))
        
        self.image = self.image[top:top+height, left:left+width]
        
        if self.ts:
            print()
            print("Top, Left from self.dimensions used in getting the slice")
            print(top)
            print(left)
        self.xOffsetRel = left
        self.yOffsetRel = top
        
        return self
    
    # Canny edge detector into HoughLinesP
    def canny_edge(self):
            
        src_gray = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
    
        low_threshold = self.cannyThreshold
        img_blur = cv.blur(src_gray, (3,3))
        detected_edges = cv.Canny(img_blur,
                                  low_threshold,
                                  low_threshold*self.ratio,
                                  self.kernel_size
                                  )

        # inclusion of HoughLineP:
        self.edges = cv.HoughLinesP(detected_edges, 1, np.pi/180, 100, 
                               minLineLength=225, maxLineGap=0
                               )
        if self.ts:
            image = np.ascontiguousarray(np.array(self.image)[:, :, :3], dtype=np.uint8)
            for x1,y1,x2,y2 in self.edges[:,0]:
                cv.line(image, (x1,y1), (x2,y2), (0,255,0), 2)
                
            cv.imshow("Edges from canny_edge", image)
            cv.waitKey(0)
            
        return self
    
    def set_board_edges(self):
        """
        self.edges is of shape 7, 1, 4 due to multiple lines being detected as
        edges of the game board.
        [x, y, x, y]
        """
        self.edges = self.edges.reshape(-1, 4)
        # Slice into x's and y's
        xs = np.concatenate([self.edges[:, 0], self.edges[:, 2]])  # all x1 and x2
        ys = np.concatenate([self.edges[:, 1], self.edges[:, 3]])  # all y1 and y2
        
        if self.ts:
            print()
            print(f"all x in edges: {xs}")
            print(f"all y in edges: {ys}")
        
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        
        # Edges are redifined to mark the play space
        self.topEdge = ymin + self.yOffsetRel
        self.leftEdge = xmin + self.xOffsetRel
        self.rightEdge = xmax + self.xOffsetRel
        self.botEdge = ymax + self.yOffsetRel
        
        return self
    
    def fill_background(self, pad=3):
        """
        Black out everything outside the circles, shrinking the circles slightly 
        to remove slivers. 
        pad: int, number of pixels to shrink the radius
        """
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
    
        for x, y, r in self.circles.reshape(-1, 3):
            x_rel = int(x - self.dimensions[1])
            y_rel = int(y - self.dimensions[0])
            r_shrink = max(int(r) - pad, 1)  # ensure radius is positive
            cv.circle(mask, (x_rel, y_rel), r_shrink, 255, thickness=-1)
    
        # Apply mask
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            mask_3ch = cv.merge([mask, mask, mask])
            masked_image = cv.bitwise_and(self.image, mask_3ch)
            
        else:
            masked_image = cv.bitwise_and(self.image, mask, mask=mask)
    
        self.image = masked_image
        if self.ts:
            cv.imshow("Filled Background", self.image)
            cv.waitKey(0)
        
        return self
    
    # Take the circle coords and return the dimensions of the cooresponding
    # square for OCR capture
    def get_squares(self, n = 8):
        lengthOfArray = len(self.circles)
        # Check length of array
        if lengthOfArray == n:
            #print(f"Passed into get_squares: {circleArray}")
            captureArray = np.empty((n, 4), dtype=int)
            
            for i, circle in enumerate(self.circles):
                # pull out useful dimensions of circle
                x = circle[0]
                y = circle[1]
                r = circle[2]

                # calculate dimensions of respective square
                left = x - r
                top = y - r
                side = r*2
                
                captureArray[i] = [left, top, side, side]

            self.squares = captureArray
            return self
        
        else:
            raise ValueError(f"Expected only 8 circles, received {lengthOfArray}")
            
            
    def extract_square_images(self):
        self.ocr_captures = [
            self.image[
                int(max(0, top - self.dimensions[0])) : int(min(height + top - self.dimensions[0], self.image.shape[0])),
                int(max(0, left - self.dimensions[1])) : int(min(width + left - self.dimensions[1], self.image.shape[1]))
                    ] 
                for left, top, width, height in self.squares
            ]
        
        if self.ts:
            for i, crop_img in enumerate(self.ocr_captures):
                image = np.ascontiguousarray(crop_img, dtype=np.uint8)
                #cv.imshow(f"Square #{i+1}", image)
                cv.waitKey(0)
            cv.destroyAllWindows()
        
        return self
    
    # preprocessing helper function for predict_digits
    def _preprocess_image(self, square_img):
            # Convert to grayscale
            gray = (cv.cvtColor(square_img, cv.COLOR_RGB2GRAY) if 
            square_img.ndim == 3 else square_img)
        
            # Apply strict threshold: everything below threshold becomes 0 (black)
            _, thresh = cv.threshold(gray, self.bin_thresh, 255, cv.THRESH_BINARY)  
        
            # Resize to MNIST input size
            resized = cv.resize(thresh, (28, 28), interpolation=cv.INTER_AREA)
            
            return resized
    
    def predict_digits(self):
        if not self.ts:
            self.predictions = [np.argmax(
                self.model.predict(self._preprocess_image(img)[None, :, :]), 
                axis=1)[0] + 1 for img in self.ocr_captures]
        
        # self.predictions is not currently updated when ts_mode is active
        elif self.ts:
            self.predictions = []
            for i, img in enumerate(self.ocr_captures):
                prediction = np.argmax(
                    self.model.predict(self._preprocess_image(img)[None, :, :]),
                    axis = 1)[0] + 1
                
                print(f"Predicted Digit: {prediction}")
                self.predictions.append(prediction)
                cv.imshow("Image #{i}", img)
                cv.waitKey(0)
            cv.destroyAllWindows()
        
        return self
    
    def order_circles(self):
        paired = list(zip(self.predictions, self.circles))
        
        paired_sorted = sorted(paired, key=lambda x: x[0])
        
        self.orderedCircles = [circle for _, circle in paired_sorted]
        
        return self
    
    def pixels_to_grid(self):
        """
        
        """
        self.xFactor = (self.rightEdge - self.leftEdge) / 6
        self.yFactor = (self.botEdge - self.topEdge) / 6
        self.grid_locations = []
        for i, item in enumerate(self.orderedCircles):
            x = int((item[0] - self.leftEdge) / self.xFactor)
            y = int((item[1] - self.topEdge) / self.yFactor)
            
            self.grid_locations.append((x, y))
        
        return self
    
    def grid_to_pixels(self, solutionPath):
        shiftedPath = [(x + 0.5, y + 0.5) for x, y in solutionPath]
        self.pixel_coords = ([(x * self.xFactor, y * self.yFactor) 
                              for x, y in shiftedPath]
                             )
        self.pixel_coords = ([(int(x + self.leftEdge), int(y + self.topEdge)) 
                                  for x, y in self.pixel_coords]
                                 )
        if self.ts:
            print("relative coord solution: ")
            print(self.pixel_coords)
        return self
    
    def get_absolute_coords(self):
        if self.ts:
            print()
            print("x and y absolute offsets: ")
            print(self.xOffsetAbs)
            print(self.yOffsetAbs)
            print()
            
        self.pixel_coords = ([(x + self.xOffsetAbs, y + self.yOffsetAbs) # y doesn't change
                              for x, y in self.pixel_coords])
        
        if self.ts:
            print("Absolute coord solution: ")
            print(self.pixel_coords)
            
        return self
                