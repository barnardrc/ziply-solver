import numpy as np
from solvers.dependents.dependents import get_checkpoints_vals_to_coords

class DLXNode:
    def __init__(self):
        self.left = self.right = self.up = self.down = self
        self.column = None
        self.row_id = None

class ColumnNode(DLXNode):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.size = 0

class DLXSolver:
    def __init__(self, matrix, row_ids, start_cell):
        self.solution = []
        self.row_ids = row_ids
        self.start_cell = start_cell
        self.root = self._build_links(matrix)
        
    def _build_links(self, matrix):
        cols = [ColumnNode(i) for i in range(len(matrix[0]))]
        root = ColumnNode("root")

        # link columns in a circle
        last = root
        for col in cols:
            last.right = col
            col.left = last
            last = col
        last.right = root
        root.left = last

        # build rows
        for r, row in enumerate(matrix):
            first_node = None
            for c, cell in enumerate(row):
                if cell == 1:
                    node = DLXNode()
                    node.column = cols[c]
                    node.row_id = self.row_ids[r]

                    # vertical linking
                    node.up = cols[c].up
                    node.down = cols[c]
                    cols[c].up.down = node
                    cols[c].up = node
                    cols[c].size += 1

                    # horizontal linking
                    if first_node:
                        node.left = first_node.left
                        node.right = first_node
                        first_node.left.right = node
                        first_node.left = node
                    else:
                        first_node = node
                        node.left = node.right = node
        return root

    def _cover(self, col):
        col.right.left = col.left
        col.left.right = col.right
        i = col.down
        while i != col:
            j = i.right
            while j != i:
                j.down.up = j.up
                j.up.down = j.down
                j.column.size -= 1
                j = j.right
            i = i.down

    def _uncover(self, col):
        i = col.up
        while i != col:
            j = i.left
            while j != i:
                j.column.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            i = i.up
        col.right.left = col
        col.left.right = col

    def _search(self):
        
        # Format final path solution
        if self.root.right == self.root:
            unordered_moves = [node.row_id for node in self.solution]
            move_map = {start: end for start, end in unordered_moves}
            path = [self.start_cell]
            current_cell = self.start_cell
            
            for _ in range(len(unordered_moves)):
                current_cell = move_map[current_cell]
                path.append(current_cell)
            
            return path
            
        # heuristic: choose column with fewest nodes
        col = None
        min_size = float("inf")
        j = self.root.right
        while j != self.root:
            if j.size < min_size:
                min_size = j.size
                col = j
            j = j.right

        self._cover(col)
        r = col.down
        while r != col:
            self.solution.append(r)
            j = r.right
            while j != r:
                self._cover(j.column)
                j = j.right

            result = self._search()
            if result is not None:
                return result

            # backtrack
            r = self.solution.pop()
            j = r.left
            while j != r:
                self._uncover(j.column)
                j = j.left
            r = r.down
        self._uncover(col)
        return None

    def solve(self):
        return self._search()


def build_dlx_matrix(board):
    """
    Build DLX matrix for Ziply.
    Columns:
      - cell_out (for all cells except the end point)
      - cell_in (for all cells except the start point)
      - one for each checkpoint transition
    Rows:
      - each valid move (cell1 -> cell2)
    """
    H, W = board.shape
    cells = [(r, c) for r in range(H) for c in range(W)]
    
    # 1. Find the start and end cells from the checkpoint numbers
    checkpoints = get_checkpoints_vals_to_coords(board)
    start_cell = checkpoints[0][1]
    end_cell = checkpoints[-1][1]

    # 2. Create column lists that EXCLUDE the start_in and end_out constraints
    cells_with_out_col = [cell for cell in cells if cell != end_cell]
    cells_with_in_col = [cell for cell in cells if cell != start_cell]

    # 3. Build the index dictionaries based on these filtered lists
    offset = 0
    cell_out_indices = {cell: i + offset for i, cell in enumerate(cells_with_out_col)}
    
    offset += len(cells_with_out_col)
    cell_in_indices = {cell: i + offset for i, cell in enumerate(cells_with_in_col)}
    
    col_count = len(cell_out_indices) + len(cell_in_indices)
    rows = []
    row_ids = []

    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    for r_from, c_from in cells:
        for dr, dc in moves:
            r_to, c_to = r_from + dr, c_from + dc
            if 0 <= r_to < H and 0 <= c_to < W:
                from_cell = (r_from, c_from)
                to_cell = (r_to, c_to)
                
                row = [0] * col_count

                # 4. Conditionally set the '1's for the move
                # A move out of from_cell is only a constraint if it's not the end point
                if from_cell in cell_out_indices:
                    row[cell_out_indices[from_cell]] = 1
                
                # A move into to_cell is only a constraint if it's not the start point
                if to_cell in cell_in_indices:
                    row[cell_in_indices[to_cell]] = 1

                rows.append(row)
                row_ids.append((from_cell, to_cell))

    return np.array(rows), row_ids


def solve_puzzle(board, simulationLength = None):
    matrix, row_ids = build_dlx_matrix(board)
    
    checkpoints = get_checkpoints_vals_to_coords(board)
    start_cell = checkpoints[0][1]
    
    solver = DLXSolver(matrix, row_ids, start_cell)
    result = solver.solve()
    return result
