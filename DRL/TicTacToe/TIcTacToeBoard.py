import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def make_move(self, move, player):
        if self.board[move] == 0:
            self.board[move] = player
        else:
            raise ValueError("Invalid move")

    def game_over(self):
        for i in range(3):
            # Columns
            if abs(np.sum(self.board[:, i])) == 3:
                return True
            # Rows
            if abs(np.sum(self.board[i, :])) == 3:
                return True
        if abs(np.sum(np.diagonal(self.board))) == 3 or abs(np.sum(np.diagonal(np.fliplr(self.board)))) == 3:
            return True
        if abs(np.sum(self.board)) == 1 and len(np.argwhere(self.board == 0)) == 0:
            return True
        return False

    def available_moves(self):
        return np.argwhere(self.board == 0)

    def reward(self):
        for i in range(3):
            # Columns
            if np.sum(self.board[:, i]) == 3:
                return 1
            # Rows
            if np.sum(self.board[i, :]) == 3:
                return 1
        if np.sum(np.diagonal(self.board)) == 3 or np.sum(np.diagonal(np.fliplr(self.board))) == 3:
            return 1
        if np.sum(self.board) == 1 and len(np.argwhere(self.board == 0)) == 0:
            return 0.5
        return -1