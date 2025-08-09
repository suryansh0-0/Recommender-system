
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
        """Generate user profiles with demographics and preferences"""
        print("Generating user profiles...")
        
        users = []
        for user_id in range(self.n_users):
            # Demographics
            age = np.random.choice([18, 25, 35, 45, 55, 65], p=[0.1, 0.25, 0.25, 0.2, 0.15, 0.05])
            gender = np.random.choice(['M', 'F', 'O'], p=[0.45, 0.45, 0.1])
            income_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            location = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.4, 0.45, 0.15])
            
            # Behavioral characteristics
            activity_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.6, 0.2])
            price_sensitivity = np.random.beta(2, 5)  # Skewed towards price-sensitive
            
            # Primary user segment (with some overlap)
            primary_segment = np.random.choice(self.user_segments)
            
            # Category preferences (normalized)
            preferences = np.random.dirichlet(np.ones(len(self.item_categories)))
            
            # Add bias based on user segment
            segment_idx = self.user_segments.index(primary_segment)
            if segment_idx < len(self.item_categories):
                preferences[segment_idx] *= 2  # Double preference for primary category
                preferences = preferences / preferences.sum()  # Renormalize
            
            user = {
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'income_level': income_level,
                'location': location,
                'activity_level': activity_level,
                'price_sensitivity': price_sensitivity,
                'primary_segment': primary_segment,
                'category_preferences': preferences.tolist()
            }
            users.append(user)
        
        self.users_df = pd.DataFrame(users)
        print(f"Generated {len(users)} user profiles")
        return self.users_df
    
    def generate_items(self):
        """Generate item catalog with features and metadata"""
        print("Generating item catalog...")
        
        items = []
        for item_id in range(self.n_items):
            # Basic item info
            category = np.random.choice(self.item_categories)
            
            # Price based on category (some categories are more expensive)
            category_price_multipliers = {
                'Electronics': 3.0, 'Travel': 4.0, 'Home & Garden': 2.0,
                'Clothing': 1.5, 'Sports': 2.5, 'Music': 1.0,
                'Movies': 1.0, 'Books': 0.8, 'Food': 1.2, 'Health': 1.8
            }
            
            base_price = np.random.lognormal(2, 1)  # Log-normal distribution
            price = base_price * category_price_multipliers.get(category, 1.0)
            
            # Item quality/rating (correlated with price but with noise)
            quality_score = min(5.0, max(1.0, 2.5 + 0.5 * np.log(price) + np.random.normal(0, 0.5)))
            
            # Popularity (some items are naturally more popular)
            popularity = np.random.beta(2, 8)  # Skewed towards less popular items
            
            # Brand tier
            brand_tier = np.random.choice(['Premium', 'Standard', 'Budget'], p=[0.2, 0.5, 0.3])
            
            # Seasonality factor (some items are seasonal)
            seasonality = np.random.choice(['None', 'Summer', 'Winter', 'Holiday'], p=[0.6, 0.15, 0.15, 0.1])
            
            # Item features (normalized)
            n_features = 20
            item_features = np.random.randn(n_features)
            
            # Add category-specific feature bias
            category_idx = self.item_categories.index(category)
            item_features[category_idx % n_features] += 2  # Strong signal for category
            
            item = {
                'item_id': item_id,
                'category': category,
                'price': round(price, 2),
                'quality_score': round(quality_score, 2),
                'popularity': round(popularity, 3),
                'brand_tier': brand_tier,
                'seasonality': seasonality,
                'features': item_features.tolist()
            }
            items.append(item)
        
        self.items_df = pd.DataFrame(items)
        print(f"Generated {len(items)} items")
        return self.items_df
    
    def generate_interactions(self, n_interactions=50000):
        """Generate realistic user-item interactions"""
        print("Generating user interactions...")
        
        interactions = []
        interaction_id = 0
        
        # Generate interactions for each user
        for user_id in range(self.n_users):
            user_data = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
            user_prefs = np.array(user_data['category_preferences'])
            
            # Number of interactions per user (heavy-tailed distribution)
            if user_data['activity_level'] == 'High':
                n_user_interactions = np.random.poisson(50)
            elif user_data['activity_level'] == 'Medium':
                n_user_interactions = np.random.poisson(20)
            else:
                n_user_interactions = np.random.poisson(5)
            
            n_user_interactions = max(1, min(n_user_interactions, 100))  # Cap at 100
            
            # Generate interactions over time
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2024, 12, 31)
            
            for _ in range(n_user_interactions):
                # Select item based on user preferences
                item_probs = []
                for _, item in self.items_df.iterrows():
                    category_idx = self.item_categories.index(item['category'])
                    
                    # Base probability from category preference
                    prob = user_prefs[category_idx]
                    
                    # Adjust for price sensitivity
                    price_factor = 1 - user_data['price_sensitivity'] * (item['price'] / 1000)
                    prob *= max(0.1, price_factor)
                    
                    # Adjust for item popularity
                    prob *= (1 + item['popularity'])
                    
                    # Adjust for quality
                    prob *= (item['quality_score'] / 5.0)
                    
                    item_probs.append(prob)
                
                # Normalize probabilities
                item_probs = np.array(item_probs)
                item_probs = item_probs / item_probs.sum()
                
                # Select item
                selected_item_idx = np.random.choice(len(self.items_df), p=item_probs)
                selected_item = self.items_df.iloc[selected_item_idx]
                
                # Generate interaction type and rating
                interaction_types = ['view', 'click', 'purchase', 'rating']
                
                # Probability of each interaction type
                if selected_item['quality_score'] > 4.0:
                    type_probs = [0.4, 0.3, 0.2, 0.1]  # High quality -> more purchases
                else:
                    type_probs = [0.5, 0.3, 0.15, 0.05]  # Lower quality -> more views
                
                interaction_type = np.random.choice(interaction_types, p=type_probs)
                
                # Generate rating (if applicable)
                if interaction_type in ['purchase', 'rating']:
                    # Rating correlated with item quality + user satisfaction
                    base_rating = selected_item['quality_score']
                    satisfaction = np.random.normal(0, 0.5)
                    rating = max(1, min(5, round(base_rating + satisfaction)))
                else:
                    rating = None
                
                # Generate timestamp
                random_days = np.random.randint(0, (end_date - start_date).days)
                timestamp = start_date + timedelta(days=random_days)
                
                # Session context
                session_id = f"{user_id}_{timestamp.strftime('%Y%m%d_%H')}"
                
                interaction = {
                    'interaction_id': interaction_id,
                    'user_id': user_id,
                    'item_id': selected_item['item_id'],
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamp,
                    'session_id': session_id
                }
                
                interactions.append(interaction)
                interaction_id += 1
        
        self.interactions_df = pd.DataFrame(interactions)
        print(f"Generated {len(interactions)} interactions")
        return self.interactions_df
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        print("Creating user-item matrix...")
        
        # Create rating matrix (explicit feedback)
        rating_interactions = self.interactions_df[
            self.interactions_df['rating'].notna()
        ].copy()
        
        if len(rating_interactions) > 0:
            rating_matrix = rating_interactions.pivot_table(
                index='user_id',
                columns='item_id', 
                values='rating',
                fill_value=0
            )
            
            # Convert to sparse matrix
            self.rating_matrix = csr_matrix(rating_matrix.values)
            self.rating_matrix_df = rating_matrix
        
        # Create interaction matrix (implicit feedback)
        interaction_counts = self.interactions_df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
        
        interaction_matrix = interaction_counts.pivot_table(
            index='user_id',
            columns='item_id',
            values='count',
            fill_value=0
        )
        
        self.interaction_matrix = csr_matrix(interaction_matrix.values)
        self.interaction_matrix_df = interaction_matrix
        
        print(f"Rating matrix shape: {self.rating_matrix.shape if hasattr(self, 'rating_matrix') else 'N/A'}")
        print(f"Interaction matrix shape: {self.interaction_matrix.shape}")
        
        return self.interaction_matrix_df
    
    def generate_content_features(self):
        """Generate content-based features for items and users with consistent dimensions"""
        print("Generating content features...")
        
        # Define consistent feature dimensions
        feature_dim = 50  # Fixed feature dimension for both users and items
        
        # Item content features
        item_features = []
        for _, item in self.items_df.iterrows():
            # Category one-hot encoding (10 features)
            category_features = [1 if cat == item['category'] else 0 
                               for cat in self.item_categories]
            
            # Price bucket (5 features)
            price_buckets = [0] * 5
            if item['price'] < 20:
                price_buckets[0] = 1
            elif item['price'] < 50:
                price_buckets[1] = 1
            elif item['price'] < 100:
                price_buckets[2] = 1
            elif item['price'] < 500:
                price_buckets[3] = 1
            else:
                price_buckets[4] = 1
            
            # Quality bucket (5 features)
            quality_buckets = [0] * 5
            quality_idx = max(0, min(4, int(item['quality_score']) - 1))
            quality_buckets[quality_idx] = 1
            
            # Brand tier (3 features)
            brand_features = [1 if tier == item['brand_tier'] else 0 
                            for tier in ['Premium', 'Standard', 'Budget']]
            
            # Popularity and seasonality (2 features)
            popularity_feature = [item['popularity']]
            seasonality_features = [1 if item['seasonality'] != 'None' else 0]
            
            # Additional features to reach feature_dim (25 features)
            # Use subset of item['features'] or pad with zeros
            additional_features = item['features'][:25] if len(item['features']) >= 25 else item['features'] + [0] * (25 - len(item['features']))
            
            # Combine all features (10 + 5 + 5 + 3 + 1 + 1 + 25 = 50)
            features = (category_features + price_buckets + quality_buckets + 
                       brand_features + popularity_feature + seasonality_features + 
                       additional_features)
            
            # Ensure exactly feature_dim dimensions
            features = features[:feature_dim]
            if len(features) < feature_dim:
                features.extend([0] * (feature_dim - len(features)))
            
            item_features.append(features)
        
        self.item_content_features = np.array(item_features)
        
        # User profile features with same dimension
        user_features = []
        for _, user in self.users_df.iterrows():
            # Demographics (6 + 3 + 3 + 3 + 3 = 18 features)
            age_features = [0] * 6
            age_buckets = [18, 25, 35, 45, 55, 65]
            age_idx = age_buckets.index(user['age'])
            age_features[age_idx] = 1
            
            gender_features = [1 if g == user['gender'] else 0 for g in ['M', 'F', 'O']]
            income_features = [1 if i == user['income_level'] else 0 for i in ['Low', 'Medium', 'High']]
            location_features = [1 if l == user['location'] else 0 for l in ['Urban', 'Suburban', 'Rural']]
            activity_features = [1 if a == user['activity_level'] else 0 for a in ['Low', 'Medium', 'High']]
            
            # Price sensitivity (1 feature)
            price_sens_features = [user['price_sensitivity']]
            
            # Category preferences (10 features) 
            category_pref_features = user['category_preferences']
            
            # User segment features (10 features)
            segment_features = [1 if seg == user['primary_segment'] else 0 
                              for seg in self.user_segments]
            
            # Additional behavioral features to reach feature_dim (11 features)
            # Generate some synthetic behavioral features
            behavioral_features = [
                np.random.beta(2, 5),  # browsing_intensity
                np.random.beta(3, 2),  # purchase_frequency  
                np.random.beta(2, 3),  # review_frequency
                np.random.beta(4, 2),  # brand_loyalty
                np.random.beta(2, 4),  # price_comparison_tendency
                np.random.beta(3, 3),  # seasonal_shopping
                np.random.beta(2, 2),  # social_influence
                np.random.beta(3, 4),  # impulse_buying
                np.random.beta(4, 3),  # research_before_purchase
                np.random.beta(2, 3),  # early_adopter
                np.random.beta(3, 2),  # deal_seeking
            ]
            
            # Combine all features (18 + 1 + 10 + 10 + 11 = 50)
            features = (age_features + gender_features + income_features + 
                       location_features + activity_features + price_sens_features +
                       category_pref_features + segment_features + behavioral_features)
            
            # Ensure exactly feature_dim dimensions
            features = features[:feature_dim]
            if len(features) < feature_dim:
                features.extend([0] * (feature_dim - len(features)))
            
            user_features.append(features)
        
        self.user_content_features = np.array(user_features)
        
        print(f"Item content features shape: {self.item_content_features.shape}")
        print(f"User content features shape: {self.user_content_features.shape}")
        
        # Verify dimensions match
        assert self.item_content_features.shape[1] == self.user_content_features.shape[1], \
            f"Feature dimensions don't match: items={self.item_content_features.shape[1]}, users={self.user_content_features.shape[1]}"
        
        return self.item_content_features, self.user_content_features
    
    def save_data(self, filepath_prefix="recommender_data", output_dir="datasets/generated"):
        """Save all generated data to files under output_dir."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, filepath_prefix)
        print(f"Saving data with base: {base}*")

        # Save DataFrames
        self.users_df.to_csv(f"{base}_users.csv", index=False)
        self.items_df.to_csv(f"{base}_items.csv", index=False)
        self.interactions_df.to_csv(f"{base}_interactions.csv", index=False)

        # Save matrices
        np.save(f"{base}_user_item_matrix.npy", self.interaction_matrix.toarray())

        if hasattr(self, 'rating_matrix'):
            np.save(f"{base}_rating_matrix.npy", self.rating_matrix.toarray())

        # Save content features
        if hasattr(self, 'item_content_features'):
            np.save(f"{base}_item_features.npy", self.item_content_features)
            np.save(f"{base}_user_features.npy", self.user_content_features)

        # Save metadata
        metadata = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': len(self.interactions_df),
            'item_categories': self.item_categories,
            'user_segments': self.user_segments,
            'generation_date': datetime.now().isoformat()
        }

        with open(f"{base}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Data saved successfully!")
    
    def analyze_data(self):
        """Generate data analysis and visualizations"""
        print("Analyzing generated data...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # User age distribution
        self.users_df['age'].hist(bins=20, ax=axes[0,0])
        axes[0,0].set_title('User Age Distribution')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Count')
        
        # Item price distribution
        self.items_df['price'].hist(bins=50, ax=axes[0,1])
        axes[0,1].set_title('Item Price Distribution')
        axes[0,1].set_xlabel('Price ($)')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_yscale('log')
        
        # Item category distribution
        category_counts = self.items_df['category'].value_counts()
        category_counts.plot(kind='bar', ax=axes[0,2], rot=45)
        axes[0,2].set_title('Item Category Distribution')
        axes[0,2].set_ylabel('Count')
        
        # Interaction type distribution
        interaction_counts = self.interactions_df['interaction_type'].value_counts()
        interaction_counts.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Interaction Type Distribution')
        axes[1,0].set_ylabel('Count')
        
        # User activity distribution (interactions per user)
        user_activity = self.interactions_df['user_id'].value_counts()
        user_activity.hist(bins=30, ax=axes[1,1])
        axes[1,1].set_title('User Activity Distribution')
        axes[1,1].set_xlabel('Interactions per User')
        axes[1,1].set_ylabel('Count')
        
        # Item popularity distribution (interactions per item)
        item_popularity = self.interactions_df['item_id'].value_counts()
        item_popularity.hist(bins=30, ax=axes[1,2])
        axes[1,2].set_title('Item Popularity Distribution')
        axes[1,2].set_xlabel('Interactions per Item')
        axes[1,2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Data Summary ===")
        print(f"Users: {self.n_users}")
        print(f"Items: {self.n_items}")
        print(f"Total Interactions: {len(self.interactions_df)}")
        print(f"Sparsity: {(1 - len(self.interactions_df) / (self.n_users * self.n_items)) * 100:.2f}%")
        
        if hasattr(self, 'rating_matrix'):
            print(f"Explicit Ratings: {self.rating_matrix.nnz}")
            print(f"Rating Sparsity: {(1 - self.rating_matrix.nnz / (self.n_users * self.n_items)) * 100:.2f}%")
        
        print(f"Average Interactions per User: {len(self.interactions_df) / self.n_users:.2f}")
        print(f"Average Interactions per Item: {len(self.interactions_df) / self.n_items:.2f}")
    
    def get_data_for_drl(self):
        """Prepare data specifically for the DRL recommender system"""
        print("Preparing data for DRL system...")
        
        # Create user-item preference matrix (ground truth for environment)
        user_features = self.user_content_features
        item_features = self.item_content_features
        
        # Normalize features
        user_features = user_features / (np.linalg.norm(user_features, axis=1, keepdims=True) + 1e-8)
        item_features = item_features / (np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute preference matrix
        preference_matrix = np.dot(user_features, item_features.T)
        preference_matrix = 1 / (1 + np.exp(-preference_matrix))  # Sigmoid activation
        
        # Add noise from actual interactions
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
        
        # Clip to [0, 1]
        preference_matrix = np.clip(preference_matrix, 0, 1)
        
        data_package = {
            'user_features': user_features,
            'item_features': item_features,
            'preference_matrix': preference_matrix,
            'interaction_history': self.interactions_df,
            'n_users': self.n_users,
            'n_items': self.n_items
        }
        
        return data_package


# Example usage
def generate_sample_data(output_dir="datasets/generated"):
    """Generate and save sample dataset"""
    print("=== Generating Sample Recommender Dataset ===")
    
    # Create data generator
    generator = RecommenderDataGenerator(n_users=1000, n_items=500)
    
    # Generate all data
    users_df = generator.generate_users()
    items_df = generator.generate_items()
    interactions_df = generator.generate_interactions(n_interactions=30000)
    
    # Create matrices
    interaction_matrix = generator.create_user_item_matrix()
    
    # Generate content features
    item_features, user_features = generator.generate_content_features()
    
    # Save data
    generator.save_data("sample_recommender_data", output_dir=output_dir)
    
    # Analyze data
    generator.analyze_data()
    
    # Get DRL-ready data
    drl_data = generator.get_data_for_drl()
    
    print("\n=== Sample Data Generated Successfully! ===")
    print("Files created:")
    print("- sample_recommender_data_users.csv")
    print("- sample_recommender_data_items.csv") 
    print("- sample_recommender_data_interactions.csv")
    print("- sample_recommender_data_user_item_matrix.npy")
    print("- sample_recommender_data_item_features.npy")
    print("- sample_recommender_data_user_features.npy")
    print("- sample_recommender_data_metadata.json")
    
    return generator, drl_data