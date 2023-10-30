import pygame
import random

from pygame import Rect

COLORS = {
    "0": ((255,255,255), (0,0,0)),
    "2": ((255,255,255), (0,0,0)),
    "4": ((210,210,255),(0,0,0)),
    "8": ((185,185,255),(0,0,0)),
    "16": ((160,160,255),(0,0,0)),
    "32": ((134,134,255),(0,0,0)),
    "64": ((108,108,255),(0,0,0)),
    "128": ((82,82,255),(255,255,255)),
    "256": ((56,56,255),(255,255,255)),
    "512": ((30,30,255),(255,255,255)),
    "1024": ((5,5,255),(255,255,255)),
    "2048": ((0,0,255),(255,255,255)),
}

class Cell:
    def __init__(self, row, col, width, val):
        self.row = row
        self.col = col
        self.x = self.row * width
        self.y = self.col * width
        self.width = width
        self.rect = Rect(self.x, self.y, self.width, self.width)
        self.val = val
        self.colors = COLORS[str(self.val)]

class GameBoard:
    def __init__(self, win, font, width):
        self.win = win
        self.font = font
        self.score = 0
        self.board_width = width
        self.cell_width = width // 4
        self.board = self.gen_board()
        self.game_over = False

    def gen_board(self):
        board = []

        start_cell = (random.randint(0, 3), random.randint(0, 3))

        for i in range(4):
            row = []
            for j in range(4):
                val = 0
                if i == start_cell[0] and j == start_cell[1]:
                    val = 2
                row.append(Cell(i, j, self.cell_width, val))
            board.append(row)
        return board

    def add_cell(self):
        val = random.choice([2, 4])
        while True:
            cell = (random.randint(0, 3), random.randint(0, 3))
            if self.board[cell[0]][cell[1]].val == 0:
                self.board[cell[0]][cell[1]].val = val
                break

    def update_colors(self):
        for row in self.board:
            for cell in row:
                cell.colors = COLORS[str(cell.val)]

# Fix shifting
    def shift_up(self):
        num_moves = 0
        moved = True
        while moved:
            moved = False
            for row in range(4):
                for col in range(3, 0, -1):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row][col - 1].val:
                        self.board[row][col - 1].val <<= 1
                        self.score += self.board[row][col - 1].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
                    if self.board[row][col].val > 0 and self.board[row][col - 1].val == 0:
                        self.board[row][col - 1].val = self.board[row][col].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
        if num_moves > 0:
            self.add_cell()
        self.update_colors()

    def shift_down(self):
        num_moves = 0
        moved = True
        while moved:
            moved = False
            for row in range(4):
                for col in range(3):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row][col + 1].val:
                        self.board[row][col + 1].val <<= 1
                        self.score += self.board[row][col + 1].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
                    if self.board[row][col].val > 0 and self.board[row][col + 1].val == 0:
                        self.board[row][col + 1].val = self.board[row][col].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
        if num_moves > 0:
            self.add_cell()
        self.update_colors()

    def shift_left(self):
        num_moves = 0
        moved = True
        while moved:
            moved = False
            for row in range(3, 0, -1):
                for col in range(4):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row - 1][col].val:
                        self.board[row - 1][col].val <<= 1
                        self.score += self.board[row - 1][col].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
                    if self.board[row][col].val > 0 and self.board[row - 1][col].val == 0:
                        self.board[row - 1][col].val = self.board[row][col].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
        if num_moves > 0:
            self.add_cell()
        self.update_colors()

    def shift_right(self):
        num_moves = 0
        moved = True
        while moved:
            moved = False
            for row in range(3):
                for col in range(4):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row + 1][col].val:
                        self.board[row + 1][col].val <<= 1
                        self.score += self.board[row + 1][col].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
                    if self.board[row][col].val > 0 and self.board[row + 1][col].val == 0:
                        self.board[row + 1][col].val = self.board[row][col].val
                        self.board[row][col].val = 0
                        moved = True
                        num_moves += 1
        if num_moves > 0:
            self.add_cell()
        self.update_colors()


    def check_up(self):
        moved = False
        for row in range(4):
            for col in range(3, 0, -1):
                if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row][col - 1].val:
                    moved = True
                if self.board[row][col].val > 0 and self.board[row][col - 1].val == 0:
                    moved = True
        return moved

    def check_down(self):
        moved = False
        for row in range(4):
            for col in range(3):
                if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row][col + 1].val:
                    moved = True
                if self.board[row][col].val > 0 and self.board[row][col + 1].val == 0:
                    moved = True
        return moved

    def check_left(self):
        moved = False
        for row in range(3, 0, -1):
            for col in range(4):
                if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row - 1][col].val:
                    moved = True
                if self.board[row][col].val > 0 and self.board[row - 1][col].val == 0:
                    moved = True
        return moved

    def check_right(self):
        moved = False
        for row in range(3):
            for col in range(4):
                if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row + 1][col].val:
                    moved = True
                if self.board[row][col].val > 0 and self.board[row + 1][col].val == 0:
                    moved = True
        return moved


    def get_vals(self):
        return [[cell.val for cell in row] for row in self.board]

    def reset(self):
        self.board = self.gen_board()
        self.score = 0
        self.game_over = False

    def check_loss(self):
        # Check for no zeros
        zero = False
        for row in self.board:
            for cell in row:
                if cell.val == 0:
                    zero = True
                    break
            if zero:
                break

        if not zero:
            num_moves = 0

            # Check up
            for row in range(4):
                for col in range(3, 0, -1):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row][col - 1].val:
                        num_moves += 1
            # Check down
            for row in range(4):
                for col in range(3):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row][col + 1].val:
                        num_moves += 1
            # Check left
            for row in range(3, 0, -1):
                for col in range(4):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row - 1][col].val:
                        num_moves += 1
            # Check right
            for row in range(3):
                for col in range(4):
                    if self.board[row][col].val != 0 and self.board[row][col].val == self.board[row + 1][col].val:
                        num_moves += 1

            if num_moves == 0:
                self.game_over = True
                return True
        return False

    def draw(self):
        self.win.fill((255,255,255))
        
        # Draw cells
        for row in self.board:
            for cell in row:
                pygame.draw.rect(self.win, cell.colors[0], cell.rect)
                if cell.val > 0:
                    text = self.font.render(str(cell.val), True, cell.colors[1])
                    self.win.blit(text, (cell.x + self.cell_width / 2, cell.y + self.cell_width / 2))
        
        # Draw grid lines
        for i in range(4):
            # Draw a horizontal line at the top and bottom of each node
            pygame.draw.line(
                self.win,
                (128, 128, 128),
                (0, i * self.cell_width),
                (self.board_width, i * self.cell_width),
            )
            for j in range(4):
                # Draw a vertical line at the left and right of each node
                pygame.draw.line(
                    self.win,
                    (128, 128, 128),
                    (j * self.cell_width, 0),
                    (j * self.cell_width, self.board_width),
                )