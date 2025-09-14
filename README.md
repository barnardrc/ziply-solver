<p align="center">
  <img src="assets/demo.gif" alt="Ziply Solver Demo" width="400"/>
</p>

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

## Installation

### Using Conda
To create the environment from the provided YAML file:

```bash
conda env create -f environment.yaml
conda activate ziply-game-env
```
### Using Pip
To install dependencies from requirements.txt:

```bash
python -m venv ziply-env
```
```bash
source ziply-env/bin/activate      # macOS/Linux
```
```bash
ziply-env\Scripts\activate.bat     # Windows
```
```bash
pip install -r requirements.txt
```
If you are on MacOS, you will also need to install pyobj.
```bash
pip install -U pyobjc
```
After installation, run the program with: 
```bash 
python main.py
```
You will be prompted to click the window that contains the puzzle. Once clicked, there is a one second delay until it begins solving. Once it is finished, you can click "New Game" or refresh the webpage to continue solving puzzles indefinitely. Pressing Ctrl + C in terminal will exit the program.
