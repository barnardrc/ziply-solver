"""
Dancing Links (DLX) solver for Ziply puzzle.
This is an alternative to DFS.
"""

import numpy as np

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
    def __init__(self, matrix, row_ids):
        self.solution = []
        self.row_ids = row_ids
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
        if self.root.right == self.root:
            return [node.row_id for node in self.solution]

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
    Convert Ziply into Exact Cover matrix.
    - Constraint 1: each cell visited exactly once.
    - Constraint 2: checkpoints appear in order.
    """
    H, W = board.shape
    cells = [(r, c) for r in range(H) for c in range(W)]
    checkpoints = sorted([board[r, c] for r, c in cells if board[r, c] > 0])

    col_count = len(cells) + len(checkpoints) - 1  # cells + ordering constraints
    rows = []
    row_ids = []

    # For each possible move (r1,c1) -> (r2,c2), build row
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    for r, c in cells:
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                row = [0] * col_count
                idx1 = r * W + c
                idx2 = nr * W + nc
                row[idx1] = 1
                row[idx2] = 1

                # if move satisfies checkpoint order
                cp1, cp2 = board[r, c], board[nr, nc]
                if cp1 > 0 and cp2 > 0:
                    if cp2 == cp1 + 1:
                        row[len(cells) + cp1 - 1] = 1
                rows.append(row)
                row_ids.append(((r, c), (nr, nc)))

    return np.array(rows), row_ids

def solve_puzzle(board, max_steps=1000):
    matrix, row_ids = build_dlx_matrix(board)
    solver = DLXSolver(matrix, row_ids)
    result = solver.solve()
    return result
