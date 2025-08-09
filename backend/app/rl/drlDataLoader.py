import numpy as np
from .dummyDataCreator import RecommenderDataGenerator


class DRLDataAdapter:
    """Adapter to use generated data with the DRL recommender system"""

    def __init__(self, data_package):
        self.data = data_package

    def create_enhanced_environment(self):
        """Create an enhanced environment using real data"""

        class EnhancedRecommenderEnvironment:
            def __init__(self, data_package):
                self.n_users = data_package['n_users']
                self.n_items = data_package['n_items']
                self.n_features = data_package['user_features'].shape[1]

                self.user_features = data_package['user_features']
                self.item_features = data_package['item_features']
                self.user_item_preferences = data_package['preference_matrix']

                # Interaction history for context
                self.interaction_history_data = data_package['interaction_history']

                # Current state
                self.current_user = 0
                self.interaction_history = []
                self.session_length = 0
                self.max_session_length = 10

            def reset(self):
                self.current_user = np.random.randint(0, self.n_users)
                self.interaction_history = []
                self.session_length = 0
                return self.get_state()

            def get_state(self):
                user_feat = self.user_features[self.current_user]

                # Add interaction history features
                history_feat = np.zeros(self.n_features)
                if len(self.interaction_history) > 0:
                    recent_items = self.interaction_history[-3:]
                    for item_id in recent_items:
                        if item_id < len(self.item_features):
                            history_feat += self.item_features[item_id]
                    history_feat /= len(recent_items)

                # Session context
                context_feat = np.array([
                    self.session_length / self.max_session_length,
                    len(self.interaction_history) / 10.0
                ])

                state = np.concatenate([user_feat, history_feat, context_feat])
                return state

            def step(self, action):
                item_id = action

                # Get reward from preference matrix
                base_reward = self.user_item_preferences[self.current_user, item_id]

                # Add realistic noise
                noise = np.random.normal(0, 0.1)
                engagement_prob = np.clip(base_reward + noise, 0, 1)

                engaged = np.random.random() < engagement_prob
                reward = 1.0 if engaged else -0.1

                # Diversity bonus
                if item_id in self.interaction_history[-3:]:
                    reward -= 0.2

                if engaged:
                    self.interaction_history.append(item_id)

                self.session_length += 1
                done = self.session_length >= self.max_session_length

                next_state = self.get_state()
                return next_state, reward, done

            def get_state_dim(self):
                return self.n_features * 2 + 2

        return EnhancedRecommenderEnvironment(self.data)
