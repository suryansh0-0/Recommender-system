from dqnRecommender import DQNRecommender
from recommenderEnvironment import RecommenderEnvironment
from drlDataLoader import DRLDataAdapter
from dummyDataCreator import generate_sample_data
import matplotlib.pyplot as plt
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
                # Choose and take action
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Train the agent
            self.agent.replay()
            
            # Update target network periodically
            if episode % 50 == 0:
                self.agent.update_target_network()
            
            # Track metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            # Callback for live progress
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
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        print("Training completed!")
    
    def evaluate(self, episodes=100):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Disable exploration
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        # Restore epsilon
        self.agent.epsilon = original_epsilon
        
        print(f"Evaluation Results:")
        print(f"Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Average Session Length: {np.mean(eval_lengths):.2f} ± {np.std(eval_lengths):.2f}")
        
        return eval_rewards, eval_lengths
    
    def plot_training_progress(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot moving average
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = [np.mean(self.episode_rewards[i-window+1:i+1]) 
                         for i in range(window-1, len(self.episode_rewards))]
            ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                    'r-', label=f'{window}-episode moving average')
            ax1.legend()
        
        # Plot session lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Session Length')
        
        plt.tight_layout()
        plt.show()
    
    def recommend_for_user(self, user_id, n_recommendations=5):
        """Generate recommendations for a specific user"""
        # Set environment to specific user
        self.env.current_user = user_id
        self.env.interaction_history = []
        self.env.session_length = 0
        
        state = self.env.get_state()
        
        # Disable exploration
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        recommendations = []
        for _ in range(n_recommendations):
            action = self.agent.act(state)
            recommendations.append(action)
            
            # Update state with recommendation
            next_state, _, _ = self.env.step(action)
            state = next_state
        
        # Restore epsilon
        self.agent.epsilon = original_epsilon
        
        return recommendations


if __name__ == "__main__":
    # Create and train the recommender system
    trainer = RecommenderTrainer(n_users=500, n_items=100)
    
    generator, drl_data = generate_sample_data()

    # Use with your DRL recommender
    adapter = DRLDataAdapter(drl_data)
    enhanced_env = adapter.create_enhanced_environment()
    trainer.set_environment(enhanced_env)
    
    # Train the model
    trainer.train(episodes=500)
    
    # Evaluate performance
    eval_rewards, eval_lengths = trainer.evaluate(episodes=50)
    
    # Plot training progress
    trainer.plot_training_progress()
    
    # Generate recommendations for a specific user
    user_id = 42
    recommendations = trainer.recommend_for_user(user_id, n_recommendations=5)
    print(f"\nRecommendations for User {user_id}: {recommendations}")
    
    # Compare with random baseline
    print("\nComparing with random baseline...")
    random_rewards = []
    for _ in range(50):
        state = trainer.env.reset()
        total_reward = 0
        while True:
            action = np.random.randint(0, trainer.env.n_items)
            next_state, reward, done = trainer.env.step(action)
            total_reward += reward
            if done:
                break
        random_rewards.append(total_reward)
    
    print(f"DQN Average Reward: {np.mean(eval_rewards):.2f}")
    print(f"Random Baseline Reward: {np.mean(random_rewards):.2f}")
    print(f"Improvement: {(np.mean(eval_rewards) - np.mean(random_rewards)):.2f}")