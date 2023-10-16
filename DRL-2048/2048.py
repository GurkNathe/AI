import pygame
from GameBoard import GameBoard

def main():
    width = 800
    pygame.init()
    win = pygame.display.set_mode((width, width))
    font = pygame.font.Font(None, 36)
    pygame.display.set_caption("2048")

    run = True

    board = GameBoard(win, font, width)

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

        board.draw()

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()