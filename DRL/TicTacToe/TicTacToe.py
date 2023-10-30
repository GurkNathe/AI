from TicTacDriver import Driver
import sys

agent = Driver()
opponent = Driver()

for i in range(int(sys.argv[1])):
    print(f"Game {i}")
    agent.play_game(opponent)