# ziply-solver
An automated solver for the Ziply puzzle game using computer vision and depth-first search.

## Description
`ziply-solver` automatically plays and solves the Ziply browser puzzle. It works by capturing the game board from the screen, detecting puzzle elements, reconstructing the board internally, and using a recursive depth-first search solver to find the valid solution path. The solution is then executed in real time using `pyautogui`.

### Processing pipeline
1. Capture the active puzzle window  
2. Detect all circles in the window  
3. Identify the cluster representing the puzzle  
4. Zoom into the puzzle region of interest (ROI)  
5. Detect puzzle board edges  
6. Extract and preprocess checkpoints  
7. Recognize checkpoint digits (OCR) and order them  
8. Convert pixel coordinates → grid coordinates  

### Solving
- The board is simulated as a 2D array.  
- A depth-first search with backtracking finds the valid path through all checkpoints.  
- Grid coordinates are mapped back to pixel coordinates.  
- `pyautogui` interacts with the browser to draw the solution automatically.  
![Demo GIF]
