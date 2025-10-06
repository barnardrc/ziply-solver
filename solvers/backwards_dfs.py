"""
Backtracking DFS solver for Ziply puzzle.
Now works backwards from the last checkpoint to the first (e.g. 8 → 1).
"""

import numpy as np

# Movement deltas (up, down, left, right)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def solve_puzzle(board: np.ndarray, max_steps: int = 1000):
    """
    Solve Ziply puzzle by DFS, working backwards from the last checkpoint to 1.

    Args:
        board (np.ndarray): The game board with checkpoints labeled 1..N.
        max_steps (int): Maximum depth to explore before giving up.

    Returns:
        list[(int, int)] | None: Path as list of coordinates, or None if no solution.
    """
    # find all checkpoints
    checkpoints = sorted(
        [board[r, c] for r in range(board.shape[0]) for c in range(board.shape[1]) if board[r, c] > 0]
    )
    last_cp = max(checkpoints)
    first_cp = min(checkpoints)

    # locate starting coordinate (the last checkpoint, e.g. 8)
    start = None
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            if board[r, c] == last_cp:
                start = (r, c)
                break
        if start: break

    visited = set()
    path = []

    def dfs(pos, current_cp):
        if len(path) > max_steps:
            return None

        r, c = pos
        visited.add(pos)
        path.append(pos)

        # reached the first checkpoint → solved
        if board[r, c] == first_cp:
            return path.copy()

        # if we’re on a checkpoint, next expected is one lower
        next_cp = current_cp - 1 if board[r, c] == current_cp else current_cp

        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if (nr, nc) not in visited:
                    # valid move if empty OR the next checkpoint in sequence
                    if board[nr, nc] == 0 or board[nr, nc] == next_cp:
                        result = dfs((nr, nc), next_cp)
                        if result is not None:
                            return result

        visited.remove(pos)
        path.pop()
        return None

    return dfs(start, last_cp)
