import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import random
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class RecommenderDataGenerator:
    """Generate realistic dummy data for recommender system training"""

    def __init__(self, n_users=2000, n_items=1000, random_seed=42):
        self.n_users = n_users
        self.n_items = n_items
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define item categories and user segments
        self.item_categories = [
            'Electronics', 'Books', 'Clothing', 'Home & Garden',
            'Sports', 'Music', 'Movies', 'Food', 'Health', 'Travel'
        ]

        self.user_segments = [
            'Tech Enthusiast', 'Book Lover', 'Fashion Forward', 'Home Decorator',
            'Fitness Guru', 'Music Fan', 'Movie Buff', 'Foodie',
            'Health Conscious', 'Travel Explorer'
        ]

    def generate_users(self):
        users = []
        for user_id in range(self.n_users):
            age = np.random.choice([18, 25, 35, 45, 55, 65], p=[0.1, 0.25, 0.25, 0.2, 0.15, 0.05])
            gender = np.random.choice(['M', 'F', 'O'], p=[0.45, 0.45, 0.1])
            income_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            location = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.4, 0.45, 0.15])
            activity_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.6, 0.2])
            price_sensitivity = np.random.beta(2, 5)
            primary_segment = np.random.choice(self.user_segments)
            preferences = np.random.dirichlet(np.ones(len(self.item_categories)))
            segment_idx = self.user_segments.index(primary_segment)
            if segment_idx < len(self.item_categories):
                preferences[segment_idx] *= 2
                preferences = preferences / preferences.sum()
            users.append({
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'income_level': income_level,
                'location': location,
                'activity_level': activity_level,
                'price_sensitivity': price_sensitivity,
                'primary_segment': primary_segment,
                'category_preferences': preferences.tolist()
            })
        self.users_df = pd.DataFrame(users)
        return self.users_df

    def generate_items(self):
        items = []
        for item_id in range(self.n_items):
            category = np.random.choice(self.item_categories)
            category_price_multipliers = {
                'Electronics': 3.0, 'Travel': 4.0, 'Home & Garden': 2.0,
                'Clothing': 1.5, 'Sports': 2.5, 'Music': 1.0,
                'Movies': 1.0, 'Books': 0.8, 'Food': 1.2, 'Health': 1.8
            }
            base_price = np.random.lognormal(2, 1)
            price = base_price * category_price_multipliers.get(category, 1.0)
            quality_score = min(5.0, max(1.0, 2.5 + 0.5 * np.log(price) + np.random.normal(0, 0.5)))
            popularity = np.random.beta(2, 8)
            brand_tier = np.random.choice(['Premium', 'Standard', 'Budget'], p=[0.2, 0.5, 0.3])
            seasonality = np.random.choice(['None', 'Summer', 'Winter', 'Holiday'], p=[0.6, 0.15, 0.15, 0.1])
            n_features = 20
            item_features = np.random.randn(n_features)
            category_idx = self.item_categories.index(category)
            item_features[category_idx % n_features] += 2
            items.append({
                'item_id': item_id,
                'category': category,
                'price': round(price, 2),
                'quality_score': round(quality_score, 2),
                'popularity': round(popularity, 3),
                'brand_tier': brand_tier,
                'seasonality': seasonality,
                'features': item_features.tolist()
            })
        self.items_df = pd.DataFrame(items)
        return self.items_df

    def generate_interactions(self, n_interactions=50000):
        interactions = []
        interaction_id = 0
        for user_id in range(self.n_users):
            user_data = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
            user_prefs = np.array(user_data['category_preferences'])
            if user_data['activity_level'] == 'High':
                n_user_interactions = np.random.poisson(50)
            elif user_data['activity_level'] == 'Medium':
                n_user_interactions = np.random.poisson(20)
            else:
                n_user_interactions = np.random.poisson(5)
            n_user_interactions = max(1, min(n_user_interactions, 100))
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2024, 12, 31)
            for _ in range(n_user_interactions):
                item_probs = []
                for _, item in self.items_df.iterrows():
                    category_idx = self.item_categories.index(item['category'])
                    prob = user_prefs[category_idx]
                    price_factor = 1 - user_data['price_sensitivity'] * (item['price'] / 1000)
                    prob *= max(0.1, price_factor)
                    prob *= (1 + item['popularity'])
                    prob *= (item['quality_score'] / 5.0)
                    item_probs.append(prob)
                item_probs = np.array(item_probs)
                item_probs = item_probs / item_probs.sum()
                selected_item_idx = np.random.choice(len(self.items_df), p=item_probs)
                selected_item = self.items_df.iloc[selected_item_idx]
                interaction_types = ['view', 'click', 'purchase', 'rating']
                if selected_item['quality_score'] > 4.0:
                    type_probs = [0.4, 0.3, 0.2, 0.1]
                else:
                    type_probs = [0.5, 0.3, 0.15, 0.05]
                interaction_type = np.random.choice(interaction_types, p=type_probs)
                if interaction_type in ['purchase', 'rating']:
                    base_rating = selected_item['quality_score']
                    satisfaction = np.random.normal(0, 0.5)
                    rating = max(1, min(5, round(base_rating + satisfaction)))
                else:
                    rating = None
                random_days = np.random.randint(0, (end_date - start_date).days)
                timestamp = start_date + timedelta(days=int(random_days)) if isinstance(start_date, datetime) else start_date
                session_id = f"{user_id}_{timestamp.strftime('%Y%m%d_%H')}" if hasattr(timestamp, 'strftime') else str(user_id)
                interactions.append({
                    'interaction_id': interaction_id,
                    'user_id': user_id,
                    'item_id': selected_item['item_id'],
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamp,
                    'session_id': session_id
                })
                interaction_id += 1
        self.interactions_df = pd.DataFrame(interactions)
        return self.interactions_df

    def create_user_item_matrix(self):
        rating_interactions = self.interactions_df[self.interactions_df['rating'].notna()].copy()
        if len(rating_interactions) > 0:
            rating_matrix = rating_interactions.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
            self.rating_matrix = csr_matrix(rating_matrix.values)
            self.rating_matrix_df = rating_matrix
        interaction_counts = self.interactions_df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
        interaction_matrix = interaction_counts.pivot_table(index='user_id', columns='item_id', values='count', fill_value=0)
        self.interaction_matrix = csr_matrix(interaction_matrix.values)
        self.interaction_matrix_df = interaction_matrix
        return self.interaction_matrix_df

    def generate_content_features(self):
        feature_dim = 50
        item_features_list = []
        for _, item in self.items_df.iterrows():
            category_features = [1 if cat == item['category'] else 0 for cat in self.item_categories]
            price_buckets = [0] * 5
            price = item['price']
            if price < 20:
                price_buckets[0] = 1
            elif price < 50:
                price_buckets[1] = 1
            elif price < 100:
                price_buckets[2] = 1
            elif price < 500:
                price_buckets[3] = 1
            else:
                price_buckets[4] = 1
            quality_buckets = [0] * 5
            quality_idx = max(0, min(4, int(item['quality_score']) - 1))
            quality_buckets[quality_idx] = 1
            brand_features = [1 if tier == item['brand_tier'] else 0 for tier in ['Premium', 'Standard', 'Budget']]
            popularity_feature = [item['popularity']]
            seasonality_features = [1 if item['seasonality'] != 'None' else 0]
            additional_features = item['features'][:25] if len(item['features']) >= 25 else item['features'] + [0] * (25 - len(item['features']))
            features = category_features + price_buckets + quality_buckets + brand_features + popularity_feature + seasonality_features + additional_features
            features = features[:feature_dim]
            if len(features) < feature_dim:
                features.extend([0] * (feature_dim - len(features)))
            item_features_list.append(features)
        self.item_content_features = np.array(item_features_list)

        user_features_list = []
        for _, user in self.users_df.iterrows():
            age_features = [0] * 6
            age_buckets = [18, 25, 35, 45, 55, 65]
            age_features[age_buckets.index(user['age'])] = 1
            gender_features = [1 if g == user['gender'] else 0 for g in ['M', 'F', 'O']]
            income_features = [1 if i == user['income_level'] else 0 for i in ['Low', 'Medium', 'High']]
            location_features = [1 if l == user['location'] else 0 for l in ['Urban', 'Suburban', 'Rural']]
            activity_features = [1 if a == user['activity_level'] else 0 for a in ['Low', 'Medium', 'High']]
            price_sens_features = [user['price_sensitivity']]
            category_pref_features = user['category_preferences']
            segment_features = [1 if seg == user['primary_segment'] else 0 for seg in self.user_segments]
            behavioral_features = [np.random.beta(2, 5), np.random.beta(3, 2), np.random.beta(2, 3), np.random.beta(4, 2), np.random.beta(2, 4), np.random.beta(3, 3), np.random.beta(2, 2), np.random.beta(3, 4), np.random.beta(4, 3), np.random.beta(2, 3), np.random.beta(3, 2)]
            features = age_features + gender_features + income_features + location_features + activity_features + price_sens_features + category_pref_features + segment_features + behavioral_features
            features = features[:feature_dim]
            if len(features) < feature_dim:
                features.extend([0] * (feature_dim - len(features)))
            user_features_list.append(features)
        self.user_content_features = np.array(user_features_list)
        return self.item_content_features, self.user_content_features

    def save_data(self, filepath_prefix: str = "recommender_data", output_dir: str = "datasets/generated"):
        """Save generated data under output_dir with a given prefix.
        Only saves if the respective artifacts were created in this session.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, filepath_prefix)
        print(f"Saving data with base: {base}*")

        # Save DataFrames
        if hasattr(self, 'users_df'):
            self.users_df.to_csv(f"{base}_users.csv", index=False)
        if hasattr(self, 'items_df'):
            self.items_df.to_csv(f"{base}_items.csv", index=False)
        if hasattr(self, 'interactions_df'):
            self.interactions_df.to_csv(f"{base}_interactions.csv", index=False)

        # Save matrices
        if hasattr(self, 'interaction_matrix'):
            np.save(f"{base}_user_item_matrix.npy", self.interaction_matrix.toarray())
        if hasattr(self, 'rating_matrix'):
            np.save(f"{base}_rating_matrix.npy", self.rating_matrix.toarray())

        # Save content features
        if hasattr(self, 'item_content_features'):
            np.save(f"{base}_item_features.npy", self.item_content_features)
        if hasattr(self, 'user_content_features'):
            np.save(f"{base}_user_features.npy", self.user_content_features)

        # Save metadata
        metadata = {
            'n_users': getattr(self, 'n_users', None),
            'n_items': getattr(self, 'n_items', None),
            'n_interactions': len(self.interactions_df) if hasattr(self, 'interactions_df') else None,
            'item_categories': getattr(self, 'item_categories', None),
            'user_segments': getattr(self, 'user_segments', None),
            'generation_date': datetime.now().isoformat()
        }
        with open(f"{base}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Data saved successfully!")

    def get_data_for_drl(self):
        user_features = self.user_content_features
        item_features = self.item_content_features
        user_features = user_features / (np.linalg.norm(user_features, axis=1, keepdims=True) + 1e-8)
        item_features = item_features / (np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-8)
        preference_matrix = 1 / (1 + np.exp(-np.dot(user_features, item_features.T)))
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            if interaction['interaction_type'] in ['purchase', 'rating']:
                boost = 0.2
            elif interaction['interaction_type'] == 'click':
                boost = 0.1
            else:
                boost = 0.05
            if user_id < self.n_users and item_id < self.n_items:
                preference_matrix[user_id, item_id] += boost
        preference_matrix = np.clip(preference_matrix, 0, 1)
        return {
            'user_features': user_features,
            'item_features': item_features,
            'preference_matrix': preference_matrix,
            'interaction_history': self.interactions_df,
            'n_users': self.n_users,
            'n_items': self.n_items,
        }
