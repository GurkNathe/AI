import numpy as np
import pygame

import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

class GameDriver():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(4,4)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(4)
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
                action = self.choose_action(state, epsilon)

                old_score = board.score

                match action:
                    case 0: board.shift_up()
                    case 1: board.shift_down()
                    case 2: board.shift_left()
                    case 3: board.shift_right()
                    case _: print("Error")

                reward = board.score - old_score

                self.update_network(state, action, reward, board)

                board.check_loss()
                board.draw()
                pygame.display.update()
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            averages = self.episode_data(board, i, averages, score_range)
        self.result_data(averages, n, score_range)

    def episode_data(self, board, i, averages, score_range):
        print(f"Score: {board.score}")
        averages += board.score
        score_range[0] = score_range[0] if score_range[0] < board.score and i != 0 else board.score
        score_range[1] = score_range[1] if score_range[1] > board.score else board.score
        return averages

    def result_data(self, averages, n, score_range):
        print(f"Weights: {self.model.get_weights()}")
        print(f"Average Score: {averages / n}")
        print(f"Range of Scores: {score_range[0]} - {score_range[1]}")

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def update_network(self, state, action, reward, board):
        target = reward
        if not board.game_over:
            target += 0.99 * np.max(self.model.predict(np.array(board.get_vals()), verbose=0)[0])
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)