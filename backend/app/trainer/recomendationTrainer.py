from ..rl.dqnRecommender import DQNRecommender
from ..rl.recommenderEnvironment import RecommenderEnvironment
import numpy as np


class RecommenderTrainer:
    """Main training class"""

    def __init__(self, n_users=1000, n_items=500):
        self.env = RecommenderEnvironment(n_users, n_items)
        self.agent = DQNRecommender(
            state_dim=self.env.get_state_dim(),
            action_dim=self.env.n_items
        )

        self.episode_rewards = []
        self.episode_lengths = []

    def set_environment(self, environment):
        """Replace environment and rebuild agent to match new dims."""
        self.env = environment
        self.agent = DQNRecommender(
            state_dim=self.env.get_state_dim(),
            action_dim=self.env.n_items
        )

    def train(self, episodes=1000, on_episode_end=None):
        """Train the recommender system"""
        print("Starting training...")

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            while True:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            self.agent.replay()

            if episode % 50 == 0:
                self.agent.update_target_network()

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            if on_episode_end is not None:
                try:
                    on_episode_end(
                        episode=episode,
                        total_reward=total_reward,
                        steps=steps,
                        epsilon=self.agent.epsilon,
                    )
                except Exception as callback_error:
                    print(f"on_episode_end callback error: {callback_error}")

            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(
                    f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                    f"Avg Length: {avg_length:.2f}, Epsilon: {self.agent.epsilon:.3f}"
                )

