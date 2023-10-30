import numpy as np
import pygame

import tensorflow as tf
import tensorflow.keras.layers as layers
tf.keras.utils.disable_interactive_logging()

class GameDriver():
    def __init__(self):
        self.model = tf.keras.Sequential([
            layers.Dense(16, activation="relu", input_shape=(1, 16)),
            layers.Dense(8, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(4, activation="softmax")
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self, board, n):
        averages = 0
        score_range = [0, 0]

        epsilon = 1
        epsilon_decay = 0.995
        min_epsilon = 0.01

        for i in range(n):
            board.reset()
            while not board.game_over:
                state = np.array(board.get_vals())

                old_score = board.score

                valid = False
                while not valid:
                    action = self.choose_action(state, epsilon)
                    print(type(action))
                    valid = self.check_validity(action, board)

                match action:
                    case 0: board.shift_up()
                    case 1: board.shift_down()
                    case 2: board.shift_left()
                    case 3: board.shift_right()
                    case _: print("Action error")

                reward = self.get_zeros(board) + (board.score - old_score)

                self.update_network(state, action, reward, board)

                board.check_loss()
                board.draw()
                pygame.display.update()
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            averages = self.episode_data(board, i, averages, score_range)
        self.result_data(averages, n, score_range)

    def episode_data(self, board, i, averages, score_range):
        print(f"Episode {i} - Score: {board.score}")
        averages += board.score
        score_range[0] = score_range[0] if score_range[0] < board.score and i != 0 else board.score
        score_range[1] = score_range[1] if score_range[1] > board.score else board.score
        return averages

    def result_data(self, averages, n, score_range):
        print(f"Weights: {self.model.get_weights()}")
        print(f"Average Score: {averages / n}")
        print(f"Range of Scores: {score_range[0]} - {score_range[1]}")

    def get_zeros(self, board):
        num_zeros = 0
        for row in board.board:
            for cell in row:
                if cell.val == 0:
                    num_zeros += 1
        return num_zeros
    
    def check_validity(self, action, board):
        valid = False
        match action:
            case 0: valid = board.check_up()
            case 1: valid = board.check_down()
            case 2: valid = board.check_left()
            case 3: valid = board.check_right()
            case _: print("Validity check error")
        return valid

    def choose_action(self, state, epsilon):
        formated_state = state.reshape(1, 1, -1)
        q_values = self.model.predict(formated_state, verbose=0)
        return np.argmax(q_values[0][0])

    def update_network(self, state, action, reward, board):
        target = reward
        formated_state = state.reshape(1, 1, -1)
        target_f = self.model.predict(formated_state, verbose=0)[0][0]
        target_f[action] = target
        print(target_f, formated_state)
        self.model.fit(formated_state, target_f, epochs=1, verbose=0)