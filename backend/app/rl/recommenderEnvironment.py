import numpy as np


class RecommenderEnvironment:
    """
    Simulated environment for recommendation system training.
    - State: user features + context
    - Action: item to recommend (item_id)
    - Reward: user engagement (click, rating, etc.)
    """

    def __init__(self, n_users=1000, n_items=500, n_features=20):
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features

        # Generate synthetic user and item features
        np.random.seed(42)
        self.user_features = np.random.randn(n_users, n_features)
        self.item_features = np.random.randn(n_items, n_features)

        # User preferences (ground truth for reward calculation)
        self.user_item_preferences = np.dot(self.user_features, self.item_features.T)
        self.user_item_preferences = 1 / (1 + np.exp(-self.user_item_preferences))  # Sigmoid

        # Current state
        self.current_user = 0
        self.interaction_history = []
        self.session_length = 0
        self.max_session_length = 10

    def reset(self):
        """Reset environment for new episode"""
        self.current_user = np.random.randint(0, self.n_users)
        self.interaction_history = []
        self.session_length = 0
        return self.get_state()

    def get_state(self):
        """Get current state representation"""
        user_feat = self.user_features[self.current_user]

        # Add interaction history features (last 3 interactions)
        history_feat = np.zeros(self.n_features)
        if len(self.interaction_history) > 0:
            recent_items = self.interaction_history[-3:]
            for item_id in recent_items:
                history_feat += self.item_features[item_id]
            history_feat /= len(recent_items)

        # Session context
        context_feat = np.array([
            self.session_length / self.max_session_length,
            len(self.interaction_history) / 10.0  # Normalized history length
        ])

        state = np.concatenate([user_feat, history_feat, context_feat])
        return state

    def step(self, action):
        """Take action (recommend item) and return reward"""
        item_id = action

        # Calculate reward based on user-item preference
        base_reward = self.user_item_preferences[self.current_user, item_id]

        # Add noise and apply engagement probability
        noise = np.random.normal(0, 0.1)
        engagement_prob = np.clip(base_reward + noise, 0, 1)

        # Binary engagement (clicked or not)
        engaged = np.random.random() < engagement_prob
        reward = 1.0 if engaged else -0.1

        # Diversity bonus (avoid repeated recommendations)
        if item_id in self.interaction_history[-3:]:
            reward -= 0.2

        # Update state
        if engaged:
            self.interaction_history.append(item_id)

        self.session_length += 1
        done = self.session_length >= self.max_session_length

        next_state = self.get_state()
        return next_state, reward, done

    def get_state_dim(self):
        return self.n_features * 2 + 2  # user + history + context
