import pygame
from GameBoard import GameBoard
from GameDriver import GameDriver

import sys

def main(argv: list):
    width = 800
    pygame.init()
    win = pygame.display.set_mode((width, width))
    font = pygame.font.Font(None, 36)
    pygame.display.set_caption("2048")

    run = True
    AI = False if int(argv[0]) == 0 else True

    board = GameBoard(win, font, width)

    if AI:
        agent = GameDriver()
        agent.train(board, 100)
        pygame.quit()
    else:
        while run and not board.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        board.shift_up()
                    if event.key == pygame.K_DOWN:
                        board.shift_down()
                    if event.key == pygame.K_LEFT:
                        board.shift_left()
                    if event.key == pygame.K_RIGHT:
                        board.shift_right()

            if board.check_loss():
                print("Game Over")
                print(f"Score: {board.score}")

            board.draw()

            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    main(sys.argv[1:])