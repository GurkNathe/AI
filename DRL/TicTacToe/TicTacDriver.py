import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from TIcTacToeBoard import TicTacToe

class Driver:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        self.model = Sequential([
            Dense(9, activation="relu", input_shape=(9,)),
            Dense(9, activation="linear")
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")

    def update_q_values(self, state, action, reward, next_state):
        # For Q-Learning
        # current_q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0][action]
        if next_state is not None:
            target_q_value = reward + self.discount_factor * np.max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
        else:
            target_q_value = reward

        self.model.fit(state.reshape(1, -1), np.array([target_q_value]), epochs=1, verbose=0)

    def choose_action(self, state):
        available_moves = state.available_moves()
        # if np.random.rand() > self.exploration_rate:
        #     return tuple(random.choice(available_moves))
        # else:
        q_values = self.model.predict(state.board.reshape(1, -1), verbose=0)[0]
        valid_q_values = q_values[available_moves]
        move = np.argmax(valid_q_values) if len(valid_q_values) > 1 else 0
        move = move if move < len(available_moves) else len(available_moves) - 1
        return tuple(available_moves[move])

    def print_board(self, board):
        board = [list(row) for row in board]
        
        board_str = ""
        for row in board:
            for i in range(3):
                if row[i] == 1:
                    row[i] = "X"
                elif row[i] == -1:
                    row[i] = "O"
                else:
                    row[i] = " "
            board_str += str(row) + "\n"
        print(board_str)

    def play_game(self, opponent):
        game = TicTacToe()
        while not game.game_over():
            action = self.choose_action(game)
            game.make_move(action, 1)

            if game.game_over():
                reward = game.reward()
                self.update_q_values(game.board.flatten(), action, reward, None)
            else:
                opponent_action = opponent.choose_action(game)
                game.make_move(opponent_action, -1)
                if game.game_over():
                    reward = game.reward()
                    self.update_q_values(game.board.flatten(), opponent_action, -reward, None)
                else:
                    reward = game.reward()
                    self.update_q_values(game.board.flatten(), opponent_action, -reward, game.board.flatten())
            self.exploration_rate *= (1 - self.exploration_decay_rate)
        self.print_board(game.board)
