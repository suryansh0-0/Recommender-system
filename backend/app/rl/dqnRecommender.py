from collections import deque
from tensorflow import keras
from keras import layers
import numpy as np
import random


class DQNRecommender:
    """Deep Q-Network for recommendation"""

    def __init__(self, state_dim, action_dim, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32

        # Experience replay buffer
        self.memory = deque(maxlen=10000)

        # Neural networks
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()

    def build_model(self):
        """Build the Q-network"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse'
        )
        return model

    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_dim)

        # Ensure correct dtype and shape
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.asarray([e[0] for e in batch], dtype=np.float32)
        actions = np.asarray([e[1] for e in batch], dtype=np.int64)
        rewards = np.asarray([e[2] for e in batch], dtype=np.float32)
        next_states = np.asarray([e[3] for e in batch], dtype=np.float32)
        dones = np.asarray([e[4] for e in batch], dtype=bool)

        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)

        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)

        # Calculate target Q values
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        self.q_network.fit(states, target_q_values, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
